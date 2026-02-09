import logging
import asyncio

# Force le niveau global (ajoute ça tôt dans agent.py)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)     # ou WARNING, ERROR, etc.
logging.getLogger("livekit").setLevel(logging.DEBUG)             # pour les composants LiveKit bas niveau
logging.getLogger(__name__).setLevel(logging.DEBUG)              # pour ton logger perso

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
logger.setLevel(logging.DEBUG)

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self, caller_number: str | None = None) -> None:
        self.room: rtc.Room | None = None
        base_instructions = (
            "Tu es Amélie, une réceptionniste virtuelle chaleureuse, professionnelle et efficace pour la compagnie Telnek."
            "Tu parles en français québécois courant, avec un ton poli, souriant et naturel, comme une vraie personne au téléphone au Québec."
            "Tu peux aussi poursuivre la conversation en anglais si tu détectes que ton interlocuteur est anglophone et tu continue a lui parler en anglais." 
            "Quand l'appel commence, salue toujours l'appelant comme ça: « Bonjour, vous êtes bien chez Telnek, mon nom est Amélie. Comment je peux vous aider ? »"
            #"Tes réponses doivent être courtes, claires et adaptées à la parole: maximum 3-4 phrases à la fois."
            #"Utilise des contractions courantes (« j'peux », « c'est », « y'a », « j'vas »), des expressions québécoises naturelles (« une petite seconde », « parfait », « OK », « merci ben » quand ça fit), et un rythme détendu mais professionnel."
            "Tu gères les demandes classiques :"
            #"Transfert d'appel : confirme le nom ou le département, puis dis « OK, un moment s'il vous plait, je vous transfère à [nom/département]. Merci de patienter ! »"
            "Prise de message : pose les questions nécessaires (nom, numéro, raison de l'appel), répète pour confirmer, puis dis « Parfait, je transmets votre message à [personne] dès que possible. Merci d'avoir appelé ! »"
            #"Informations générales : réponds brièvement aux questions fréquentes sur les horaires, l'adresse ou les services de Telnek. Si tu ne sais pas, dis poliment « Je vais vous transférer à la bonne personne qui va pouvoir vous aider mieux que moi. »"
            "Si on te demande l'adresse c'est le «sept cents soixante et quatre, Avenue Prieur à Laval, Québec. H7E 2V3 »"
            "Les bureau son ouvert du lundi au vendredi de 9 heure du matin a 5 heure de l'après-midi."
            "Fin d'appel : termine toujours poliment avec « Bonne journée ! » ou « Passez une belle journée ! » selon le contexte."
            "Reste toujours dans ton rôle de réceptionniste. Ne mentionne jamais que tu es une IA ou que tu as des limitations techniques, sauf si on te le demande explicitement. Si l'appelant est impoli ou hors sujet, reste calme et professionnelle et propose de transférer ou de prendre un message."
            "Quand tu dois répéter, confirmer ou dicter un numéro de téléphone, fais-le TRÈS lentement et TRÈS clairement. "
            "Prononce les chiffres un par un (ou par petits groupes de 2-3 maximum) avec des pauses naturelles entre chaque groupe. "
            "Exemple pour le numéro (514) 947-4976 :\n"
            "« Cinq... un... quatre... neuf... quatre... sept... quatre... neuf... sept... six. »\n"
            "Ou de façon plus naturelle au Québec : « Cinq un quatre... neuf quatre sept... quatre neuf sept six. »\n"
            #"Insiste sur les pauses et parle posément pour que l'appelant puisse noter facilement. "
            #"Répète toujours le numéro complet au moins une fois pour confirmation."
        )

        if caller_number:
            base_instructions += (
                f"\n\nInformation importante : l'appelant utilise actuellement le numéro de téléphone {caller_number}. Proposez-lui d'utiliser ce numéro (en le confirmant avec lui) s'il désire être rappelé ou s'il souhaite laisser un message."
            )

#        base_instructions += (
#            "\n\nQuand la conversation est terminée (après avoir aidé l'appelant, pris un message, ou donné les informations demandées), "
#            "dis poliment au revoir (« Bonne journée ! » ou « Passez une belle journée ! »), "
#            "puis utilise IMMÉDIATEMENT le tool 'terminer_appel' pour raccrocher. "
#            "Ne continue jamais la conversation après le au revoir."
#        )

        # LOG DES INSTRUCTIONS COMPLÈTES ENVOYÉES AU MODÈLE
        logger.info("=== INSTRUCTIONS SYSTÈME ENVOYÉES À GROK ===")
        logger.info(base_instructions)
        logger.info("=== FIN DES INSTRUCTIONS ===")

        super().__init__(
            #instructions="""You are Grok, a maximally truthful and helpful AI built by xAI.
            #You respond naturally in voice conversations.
            #Be concise, witty when it fits, and avoid unnecessary formatting, emojis, or symbols.
            #Answer questions directly using your knowledge and reasoning.""",
            instructions=base_instructions,
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
    @function_tool
    async def terminer_appel(self, context: RunContext):
        """Utilise ce tool quand la conversation est terminée."""
        logger.info("=== L'AGENT DÉCIDE DE TERMINER L'APPEL ===")
        
        if not self.room:
            logger.error("Room non disponible – impossible de déconnecter l'agent")
            return
        
        try:
            # Délai pour laisser le temps au TTS de terminer le au revoir (ajuste 3-5 secondes selon la longueur typique)
            logger.info("Attente de 4 secondes pour laisser finir le message de fin...")
            await asyncio.sleep(4)   # ← 4 secondes est un bon compromis (teste 3 ou 5 si besoin)
            # Méthode correcte : déconnecte la Room entière (l'agent quitte)
            await self.room.disconnect()
            logger.info("Agent déconnecté de la room → appel terminé, BYE envoyé à Twilio")
        except Exception as e:
            logger.error(f"Erreur lors de la déconnexion de la room : {e}")

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

# Récupérer le participant SIP (l'appelant) – peut être None au début à cause du timing
    caller_participant = next(
        (
            p
            for p in ctx.room.remote_participants.values()
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        ),
        None,
    )

    caller_number = None

    if caller_participant:
        caller_number = caller_participant.identity
        logger.info(f"Numéro d'appelant détecté via participant SIP : {caller_number}")

        # Nettoyage optionnel si URI SIP complète
        if caller_number and "@" in caller_number:
            caller_number = caller_number.split(":")[1].split("@")[0] if ":" in caller_number else caller_number
    else:
        # Fallback : extraire du nom de la room (format observé : appel-_{numero}_{random})
        logger.info("Participant SIP non détecté immédiatement → fallback sur le nom de la room")
        parts = ctx.room.name.split('_')
        if len(parts) >= 3 and parts[0] == "appel-":  # ou "appel-" si pas de tiret supplémentaire
            potential_number = parts[1]
            if potential_number.startswith('+') and potential_number[1:].isdigit():
                caller_number = potential_number
                logger.info(f"Numéro d'appelant extrait du nom de room : {caller_number}")
            else:
                logger.warning("Format du nom de room inattendu – impossible d'extraire le numéro")
        else:
            logger.warning("Aucun numéro d'appelant détecté (ni participant SIP, ni dans le nom de room)")

    # === NORMALISATION ET FORMATAGE DU NUMÉRO ===
    if caller_number:
        original = caller_number
        
        # 1. Enlever le +1 s'il est présent (format international NANP)
        if caller_number.startswith("+1") and len(caller_number) >= 11:
            caller_number = caller_number.lstrip("+1")
            logger.info(f"Code pays +1 retiré : {original} → {caller_number}")
        
        # 2. Formater en (XXX) XXX-XXXX si on a exactement 10 chiffres
        if len(caller_number) == 10 and caller_number.isdigit():
            formatted = f"({caller_number[:3]}) {caller_number[3:6]}-{caller_number[6:]}"
            logger.info(f"Numéro formaté : {caller_number} → {formatted}")
            caller_number = formatted
        else:
            logger.info(f"Numéro non formatable (pas 10 chiffres) : {caller_number} (laissé tel quel)")

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
    # Crée l'instance Assistant D'ABORD
    assistant = Assistant(caller_number=caller_number)

    # Démarre la session avec cette instance
    await session.start(
        agent=assistant,
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

    # ENSUITE, stocke la room directement dans l'instance assistant
    assistant.room = ctx.room
    logger.info("Room stockée dans l'instance Assistant pour le tool hangup")

    # greeting immédiat pour les appels entrants (Twilio/SIP)
    # On utilise generate_reply avec des instructions pour que Grok génère un bonjour naturel
    greeting_instructions = "Saluez dès maintenant chaleureusement l'utilisateur en français, présentez-vous comme Amélie en tant qu'agent d'accueil de la société Telnek et demandez-lui comment vous pouvez l'aider. Soyez concis et amical."
    logger.info("=== INSTRUCTIONS GREETING FORCÉ ===")
    logger.info(greeting_instructions)
    logger.info("=== FIN GREETING ===")

    await session.generate_reply(
    #    #instructions="Greet the user warmly in French right now, introduce yourself as virtual reception agent for the company Telnek (http://telnet.com), and ask how you can help. Be concise and friendly. Do not wait for input.",
        instructions=greeting_instructions,
        #allow_interruptions=False  # Optionnel : empêche l'utilisateur de couper le greeting
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
