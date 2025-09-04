import logging
import asyncio
import wave
import tempfile
import os
import json
from livekit import rtc
from livekit.agents import cli, JobContext, WorkerOptions, AutoSubscribe
from audio import predecir_llanto, load_llanto_model

TEMP_DIR = tempfile.gettempdir()
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # muestra DEBUG, INFO, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)

logger = logging.getLogger("agent")

async def process_audio(track: rtc.RemoteAudioTrack, room_name: str):
    audio_stream = rtc.AudioStream(track)
    buffer = bytearray()

    sample_rate = None
    channels = None
    sample_width = 2  # 16-bit PCM => 2 bytes
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


async def entrypoint(ctx: JobContext):
    """Función principal de cada job (se ejecuta cuando el agente entra a un room)."""

    room_name = ctx.room.name
    logger.debug(f"Agente conectado al room '{room_name}'")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.debug("Nuevo track de audio suscrito")
            asyncio.create_task(process_audio(track, room_name))

    # conectar al room (solo audio para este caso)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

def prewarm(ctx):
    load_llanto_model()

if __name__ == "__main__":
    opts = WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name='BabyWise_Agent')
    cli.run_app(opts)
