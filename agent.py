import logging
import asyncio
import wave
import tempfile
import os
from livekit import rtc
from livekit.agents import cli, JobContext, WorkerOptions, AutoSubscribe
from audio import predecir_llanto, load_llanto_model
from motion import detectar_movimiento
from api import report_detection_event

TEMP_DIR = tempfile.gettempdir()
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)

logger = logging.getLogger("agent")


class BabyWiseAgent:
    def __init__(self):
        self._latest_frame = None
        self._prev_frame = None
        self._video_stream = None
        self._tasks = []
        self._frame_counter = 0
        self._frame_interval = 10  # Analiza cada 10 frames
        self._room_name = None

    async def on_enter(self, ctx: JobContext):
        room = ctx.room
        self._room_name = room.name
        logger.debug(f"Agente conectado al room '{self._room_name}'")

        # Suscribirse a tracks existentes
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track:
                    self._handle_track(track, participant)

        # Suscribirse a nuevos tracks
        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            self._handle_track(track, participant)

        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    def _handle_track(self, track, participant=None):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.debug("Nuevo track de audio suscrito")
            asyncio.create_task(self.process_audio(track, participant, self._room_name))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.debug("Nuevo track de video suscrito")
            self._create_video_stream(track, participant)

    async def process_audio(self, track: rtc.RemoteAudioTrack, participant: rtc.RemoteParticipant, room_name: str):
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
                    logger.debug(f"🤖 [Agent] Predicción llanto: {resultado} de {participant} para el {room_name}")
                    if resultado == "cry":
                        report_detection_event(
                            group=room_name,
                            baby=participant.identity if participant else "unknown",
                            event_type="LLANTO"
                        )
                buffer = buffer[target_bytes:]
        await audio_stream.aclose()

    def _create_video_stream(self, track, participant=None):
        # Cerrar stream anterior si existe
        if self._video_stream is not None:
            self._video_stream.close()
        self._video_stream = rtc.VideoStream(track)
        task = asyncio.create_task(self._read_video_stream(participant))
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)

    async def _read_video_stream(self, participant=None):
        participant_identity = getattr(participant, 'identity', None) if participant else None
        async for event in self._video_stream:
            self._prev_frame, self._latest_frame = self._latest_frame, event.frame
            self._frame_counter += 1
            if self._prev_frame is not None and self._frame_counter % self._frame_interval == 0:
                if detectar_movimiento(self._prev_frame, self._latest_frame):
                    logger.debug(f"🤖 [Agent] Movimiento detectado! Room: {self._room_name}, Participant: {participant_identity}")
                    report_detection_event(
                        group=self._room_name,
                        baby=participant_identity,
                        event_type="MOVIMIENTO"
                    )

def prewarm(ctx):
    load_llanto_model()

async def entrypoint(ctx: JobContext):
    agent = BabyWiseAgent()
    await agent.on_enter(ctx)

if __name__ == "__main__":
    opts = WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,job_memory_warn_mb=1000, agent_name='BabyWise_Agent')
    cli.run_app(opts)
