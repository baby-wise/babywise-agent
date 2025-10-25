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
    def _cleanup_participant(self, identity):
        """Limpia recursos de streams y tareas de un participante."""
        if identity not in self._participants:
            return
        video_state = self._participants[identity].get("video", {})
        if video_state.get("_video_stream") is not None:
            video_state["_video_stream"].close()
        if "_tasks" in video_state:
            for t in video_state["_tasks"]:
                t.cancel()
        # Si se quiere limpiar audio, agregar aquí
        del self._participants[identity]
    def __init__(self):
        # Estado por participante: {identity: {"video": {...}, "audio": {...}}}
        self._participants = {}
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
        identity = getattr(participant, 'identity', None) if participant else None
        if not identity or not identity.startswith("camera-"):
            logger.debug(f"Ignorando track de participante no-cámara: {identity}")
            return
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.debug(f"Nuevo track de audio suscrito para {identity}")
            asyncio.create_task(self.process_audio(track, participant, self._room_name))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.debug(f"Nuevo track de video suscrito para {identity}")
            self._create_video_stream(track, participant)

    async def process_audio(self, track: rtc.RemoteAudioTrack, participant: rtc.RemoteParticipant, room_name: str):
        identity = getattr(participant, 'identity', None) if participant else None
        if not identity:
            return
        # Inicializar estado de audio si se requiere
        if identity not in self._participants:
            self._participants[identity] = {"video": {}, "audio": {}}
        # El buffer es local a la corrutina, pero si se quiere guardar estado persistente, usar self._participants[identity]["audio"]
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
                logger.debug(f"🎙 Audio detectado: {sample_rate} Hz, {channels} canales para {identity}")
            buffer.extend(frame.data)
            if target_bytes and len(buffer) >= target_bytes:
                logger.debug(f"Pasaron 7 segundos mandando a predecir para {identity}")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
                    with wave.open(tmp_file, "wb") as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(sample_width)
                        wf.setframerate(sample_rate)
                        wf.writeframes(buffer[:target_bytes])
                    tmp_file.flush()
                    resultado = predecir_llanto(tmp_file.name)
                    logger.debug(f"🤖 [Agent] Predicción llanto: {resultado} de {identity} para el {room_name}")
                    if resultado == "cry":
                        report_detection_event(
                            group=room_name,
                            baby=identity,
                            event_type="LLANTO"
                        )
                buffer = buffer[target_bytes:]
        await audio_stream.aclose()

    def _create_video_stream(self, track, participant=None):
        identity = getattr(participant, 'identity', None) if participant else None
        if not identity:
            return
        # Inicializar estado si no existe
        if identity not in self._participants:
            self._participants[identity] = {"video": {}, "audio": {}}
        video_state = self._participants[identity]["video"]
        # Cerrar stream anterior si existe
        if video_state.get("_video_stream") is not None:
            video_state["_video_stream"].close()
        video_state["_video_stream"] = rtc.VideoStream(track)
        video_state["_latest_frame"] = None
        video_state["_prev_frame"] = None
        video_state["_frame_counter"] = 0
        if "_tasks" not in video_state:
            video_state["_tasks"] = []
        task = asyncio.create_task(self._read_video_stream(participant))
        task.add_done_callback(lambda t: video_state["_tasks"].remove(t))
        video_state["_tasks"].append(task)

    async def _read_video_stream(self, participant=None):
        identity = getattr(participant, 'identity', None) if participant else None
        if not identity or identity not in self._participants:
            return
        video_state = self._participants[identity]["video"]
        video_stream = video_state.get("_video_stream")
        if not video_stream:
            return
        async for event in video_stream:
            video_state["_prev_frame"], video_state["_latest_frame"] = video_state.get("_latest_frame"), event.frame
            video_state["_frame_counter"] = video_state.get("_frame_counter", 0) + 1
            if video_state["_prev_frame"] is not None and video_state["_frame_counter"] % self._frame_interval == 0:
                if detectar_movimiento(video_state["_prev_frame"], video_state["_latest_frame"]):
                    logger.debug(f"🤖 [Agent] Movimiento detectado! Room: {self._room_name}, Participant: {identity}")
                    report_detection_event(
                        group=self._room_name,
                        baby=identity,
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
