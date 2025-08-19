import asyncio
import aiohttp
from aiohttp import web
import os
from dotenv import load_dotenv
from livekit import rtc
from audio import predecir_llanto
import wave
import tempfile
TEMP_DIR = tempfile.gettempdir()

# Cargar variables de entorno
load_dotenv()
active_agents = {}
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
BACKEND_URL = os.getenv("BACKEND_URL")

os.makedirs(TEMP_DIR, exist_ok=True)

async def process_audio(track: rtc.RemoteAudioTrack, room_name):
    audio_stream = rtc.AudioStream(track)

    buffer = bytearray()
    file_count = 0

    sample_rate = None
    channels = None
    sample_width = 2  # 16-bit PCM => 2 bytes

    target_bytes = None  # Lo calculamos cuando tengamos el primer frame

    async for event in audio_stream:
        frame = event.frame

        # Detectar parámetros de audio en el primer frame
        if sample_rate is None or channels is None:
            sample_rate = frame.sample_rate
            channels = frame.num_channels
            target_bytes = sample_rate * sample_width * channels * 6
            print(f"🎙 Audio detectado: {sample_rate} Hz, {channels} canales")

        buffer.extend(frame.data)

        # Guardar cada 6 segundos
        if target_bytes and len(buffer) >= target_bytes:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                with wave.open(tmp_file, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.writeframes(buffer[:target_bytes])

                tmp_file.flush()
                resultado = predecir_llanto(tmp_file.name)
                print(f"🤖 Predicción: {resultado} en el archivo: {tmp_file.name}")

            buffer = buffer[target_bytes:]

    await audio_stream.aclose()

async def monitor_participants(room, room_name):
    """Chequea periódicamente participantes y desconecta si queda solo el agente"""
    while True:
        total = len(room.remote_participants) + 1  # +1 por el agente local
        #print(f"[Monitor] Participantes actuales (incluye agente): {total}")
        if total == 1:
            print("[Monitor] Solo queda el agente, desconectando...")
            await room.disconnect()
            active_agents.pop(room_name, None)
            print(f"🛑 Agente desconectado de {room_name}")
            break
        await asyncio.sleep(5)

async def connect_agent_to_room(room_name, agent_identity):
    """Conecta un agente LiveKit a un room usando token del backend"""
    if room_name in active_agents:
        print(f"⚠️ Agente ya conectado a {room_name}")
        return

    print(f"🔄 Pidiendo token para room '{room_name}'...")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            BACKEND_URL,
            params={"roomName": room_name, "participantName": agent_identity},
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Error pidiendo token al backend: {resp.status}")
            data = await resp.json()
            token = data.get("token")

    if not token:
        print("❌ No se recibió token.")
        return

    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print("🔔 Nuevo track de audio suscrito.")
            asyncio.create_task(process_audio(track, room_name))

    try:
        await room.connect(LIVEKIT_URL, token)

        print(f"✅ Conectado al room '{room_name}' como '{agent_identity}'")

        # Lanzar monitoreo de participantes en background
        asyncio.create_task(monitor_participants(room, room_name))

        # Mantener conexión abierta mientras el room exista
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"❌ Error conectando al room {room_name}: {e}")
    finally:
        # Siempre se ejecuta, desconectamos y limpiamos
        if room_name in active_agents:
            await room.disconnect()
            active_agents.pop(room_name, None)
            print(f"🛑 Agente desconectado de {room_name}")


async def handle_new_room(request):
    """Endpoint HTTP que recibe el nombre del room y lanza el agente"""
    data = await request.json()
    room_name = data.get("roomName")
    agent_identity = data.get("agentName", "python-agent")

    if not room_name:
        return web.json_response({"error": "roomName es requerido"}, status=400)

    asyncio.create_task(connect_agent_to_room(room_name, agent_identity))
    return web.json_response({"status": "agente iniciado", "room": room_name})


async def start_server():
    app = web.Application()
    app.router.add_post("/spawnAgent", handle_new_room)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 5000)
    await site.start()
    print("🚀 Worker escuchando")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(start_server())
