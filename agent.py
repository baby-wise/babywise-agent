import logging
import asyncio
import wave
import tempfile
import os
from livekit import rtc
from livekit.agents import cli, JobContext, WorkerOptions, AutoSubscribe
from audio import predecir_llanto, load_llanto_model
from motion import detectar_movimiento

TEMP_DIR = tempfile.gettempdir()
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.WARNING)

logger = logging.getLogger("agent")


class BabyWiseAgent:
    def __init__(self):
        self._latest_frame = None
        self._prev_frame = None
        self._video_stream = None
        self._tasks = []

    async def on_enter(self, ctx: JobContext):
        room = ctx.room
        room_name = room.name
        logger.debug(f"Agente conectado al room '{room_name}'")

        # Suscribirse a tracks existentes
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track:
                    self._handle_track(track, room_name)

        # Suscribirse a nuevos tracks
        @room.on("track_subscribed")
        def on_track_subscribed(track, *_):
            self._handle_track(track, room_name)

        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    def _handle_track(self, track, room_name):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.debug("Nuevo track de audio suscrito")
            asyncio.create_task(self.process_audio(track, room_name))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.debug("Nuevo track de video suscrito")
            self._create_video_stream(track)

    async def process_audio(self, track: rtc.RemoteAudioTrack, room_name: str):
        audio_stream = rtc.AudioStream(track)
        buffer = bytearray()
        sample_rate = None
        channels = None
        sample_width = 2
        target_bytes = None

        async for event in audio_stream:
            frame = event.frame
            if sample_rate is None or channels is None:
                sample_rate = frame.sample_rate
                channels = frame.num_channels
                target_bytes = sample_rate * sample_width * channels * 7
                logger.debug(f"🎙 Audio detectado: {sample_rate} Hz, {channels} canales")
            buffer.extend(frame.data)
            if target_bytes and len(buffer) >= target_bytes:
                logger.debug("Pasaron 7 segundos mandando a predecir")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                    with wave.open(tmp_file, "wb") as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(sample_width)
                        wf.setframerate(sample_rate)
                        wf.writeframes(buffer[:target_bytes])
                    tmp_file.flush()
                    resultado = predecir_llanto(tmp_file.name)
                    logger.debug(f"🤖 Predicción: {resultado} en {tmp_file.name} para el {room_name}")
                buffer = buffer[target_bytes:]
        await audio_stream.aclose()

    def _create_video_stream(self, track):
        # Cerrar stream anterior si existe
        if self._video_stream is not None:
            self._video_stream.close()
        self._video_stream = rtc.VideoStream(track)
        task = asyncio.create_task(self._read_video_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)

    async def _read_video_stream(self):
        async for event in self._video_stream:
            self._prev_frame, self._latest_frame = self._latest_frame, event.frame
            if self._prev_frame is not None:
                detectar_movimiento(self._prev_frame, self._latest_frame)

def prewarm(ctx):
    load_llanto_model()

async def entrypoint(ctx: JobContext):
    agent = BabyWiseAgent()
    await agent.on_enter(ctx)

if __name__ == "__main__":
    opts = WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, job_memory_warn_mb=0, agent_name='BabyWise_Agent')
    cli.run_app(opts)
