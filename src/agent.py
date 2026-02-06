import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext,
)
from livekit.plugins import noise_cancellation, silero, xai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Grok, a maximally truthful and helpful AI built by xAI.
            You respond naturally in voice conversations.
            Be concise, witty when it fits, and avoid unnecessary formatting, emojis, or symbols.
            Answer questions directly using your knowledge and reasoning.""",
            llm=xai.realtime.RealtimeModel(voice="ara"),
            tools=[
                xai.realtime.XSearch(),         # search X (Twitter) in realtime
                xai.realtime.WebSearch(),       # general web search
                # your own @function_tool decorated methods here
            ],
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        #stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        #llm=inference.LLM(model="openai/gpt-4.1-mini"),
        llm=xai.realtime.RealtimeModel(
            voice="ara",                # default voice; "ara", others listed
            # Optional: custom turn detection (server VAD is used by default)
            # turn_detection=None,            # to disable built-in turn detection
            # or customize:
            # turn_detection=turn_detection.ServerVad(
            #     threshold=0.5,
            #     silence_duration_ms=250,
            #     prefix_padding_ms=300,
            # ),
        ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        #tts=inference.TTS(
        #    model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        #),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        #turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # greeting immédiat pour les appels entrants (Twilio/SIP)
    # On utilise generate_reply avec des instructions pour que Grok génère un bonjour naturel
    await session.generate_reply(
    #    #instructions="Greet the user warmly in French right now, introduce yourself as virtual reception agent for the company Telnek (http://telnet.com), and ask how you can help. Be concise and friendly. Do not wait for input.",
        instructions="Saluez dès maintenant chaleureusement l'utilisateur en français, présentez-vous comme Amélie en tant qu'agent d'accueil de la société Telnek et demandez-lui comment vous pouvez l'aider. Soyez concis et amical.",
        allow_interruptions=False  # Optionnel : empêche l'utilisateur de couper le greeting
    )

    # Option alternative plus simple (texte fixe, sans passer par le LLM) :
    # await session.say(
    #     "Bonjour ! Je suis Grok, comment puis-je vous aider aujourd'hui ?",
    #     allow_interruptions=False
    # )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
