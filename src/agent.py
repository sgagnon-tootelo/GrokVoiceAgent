import logging
import asyncio
import os

from twilio.rest import Client
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from livekit import (rtc, api)
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
    get_job_context,
)
from livekit.plugins import noise_cancellation, silero, xai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import deepgram
#from livekit.agents import Worker, WorkerOptions

from typing import Optional

# Force le niveau global (ajoute ça tôt dans agent.py)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)      # ou WARNING, ERROR, etc.
logging.getLogger("livekit").setLevel(logging.DEBUG)             # pour les composants LiveKit bas niveau
logging.getLogger(__name__).setLevel(logging.DEBUG)              # pour ton logger perso

logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)

load_dotenv(".env.local")

def format_phone(number: str) -> str:
    number = ''.join(filter(str.isdigit, number))
    if len(number) == 10:
        return f"({number[:3]}) {number[3:6]}-{number[6:]}"
    elif len(number) == 11 and number.startswith('1'):
        return f"({number[1:4]}) {number[4:7]}-{number[7:]}"
    return number

# Charge les vars personnalisées depuis le .env (avec fallback Telnek pour tes tests)
agent_name = os.getenv("AGENT_NAME", "Amélie")

class Assistant(Agent):
    def __init__(
            self, 
            caller_number: Optional[str] = None,
            company_name: str = "Telnek",
            company_address: str = "",
            company_hours: str = "",
            admin_phone: str = ""
            ) -> None:
        self.room: rtc.Room | None = None
        self.admin_phone = admin_phone

        logger.debug(f"AGENT_NAME: {agent_name}")
        logger.debug(f"COMPANY_NAME: {company_name}")
        logger.debug(f"COMPANY_ADDRESS: {company_address}")
        logger.debug(f"COMPANY_HOURS: {company_hours}")
        logger.debug(f"admin_phone: {admin_phone}")

        base_instructions = (
            f"Tu es {agent_name}, une réceptionniste virtuelle chaleureuse, professionnelle et efficace pour la compagnie {company_name}.\n"
            f"Tu parles en français québécois courant, avec un ton poli, souriant et naturel, comme une vraie personne au téléphone au Québec.\n"
            f"Tes réponses doivent être courtes et naturelles : maximum 2-3 phrases à la fois. Parle à un rythme détendu, avec des pauses naturelles comme une vraie personne. Utilise des contractions courantes du français québécois (« j’peux », « c’est », « y’a », « j’vas », « laissez-moi »), des petites expressions chaleureuses (« une petite seconde », « parfait », « OK », « merci ben » quand ça fit), mais reste toujours polie et professionnelle.\n"
            f"Toujours vouvoyer l’appelant : utilise « vous », « laissez-moi », « pourriez-vous », etc. Évite complètement le tutoiement et les expressions trop familières comme « bein » (dis plutôt « bien »). Reste chaleureuse mais professionnelle.\n"
            f"Tu peux aussi poursuivre la conversation en anglais si tu détectes que ton interlocuteur est anglophone et tu continue a lui parler en anglais tout le reste de l'appel.\n" 
            f"Quand l'appel commence, salue toujours l'appelant comme ça: « Bonjour, vous êtes bien chez {company_name}, mon nom est {agent_name}. Comment je peux vous aider ? »\n"
            f"Tu gères les demandes classiques :\n"
            #f"Transfert d'appel : confirme le nom ou le département, puis dis « OK, un moment s'il vous plait, je vous transfère à [nom/département]. Merci de patienter ! »\n"
            f"Prise de message ou rendez-vous :\n"
            f"- Commence par demander la personne recherchée ou le département.\n"
            f"- Ensuite, demande le sujet ou la raison de l'appel (une seule question).\n"
            f"- Propose d'abord d'utiliser le numéro actuel pour le rappel : « Je peux utiliser le numéro d'où vous appelez, qui est le [numéro formaté lentement], ou préférez-vous m'en donner un autre ? »\n"
            f"- Si l'appelant confirme le numéro actuel ou en donne un autre, note-le sans répéter inutilement.\n"
            f"- Demande le nom complet seulement quand c'est nécessaire, et toujours séparément.\n"
            f"- Une fois toutes les infos recueillies, répète UNE SEULE FOIS pour confirmation : « Juste pour confirmer : [nom], [numéro], [message/sujet]. C’est bien ça ? »\n"
            f"- Pose toujours UNE SEULE question ou demande à la fois. Attends la réponse complète de l’appelant avant de continuer. Progresse étape par étape, calmement.\n"
            f"- Une fois confirmé, appelle IMMÉDIATEMENT le tool take_message avec les paramètres exacts (name, callback_number, reason).\n"
            f"- Ne jamais appeler take_message avant d’avoir reçu une confirmation explicite de l’appelant après le récapitulatif.\n"
            f"- APRÈS avoir appelé take_message, dis EXACTEMENT cette phrase finale comme dernière réponse : « Parfait, je transmets votre message dès que possible. Merci d'avoir appelé ! Passez une belle journée ! »\n"
            f"- Parle cette phrase calmement et chaleureusement, avec une pause naturelle à la fin.\n"
            f"- IMMÉDIATEMENT après avoir fini de dire cette phrase (et seulement après), appelle le tool end_call pour terminer l'appel.\n"
            f"- Ne dis RIEN d'autre. Ne pose plus de question. Ne relance pas.\n"
            f"- CRUCIAL : Tu NE DOIS JAMAIS appeler le tool take_message avant d’avoir entendu et reçu une confirmation EXPLICITE de l’appelant APRÈS le récapitulatif (ex. « oui », « c’est correct », « parfait », « c’est ça »).\n"
            f"- Si tu n’as pas encore la confirmation, tu NE FAIS RIEN et tu ATTENDS silencieusement la réponse.\n"
            f"- Ne anticipe JAMAIS la confirmation. Même si tout semble complet, attends toujours la réponse verbale.\n"
            f"- Exemple strict de flux :\n"
            f"  - Tu poses le récapitulatif : « Juste pour confirmer : [nom], [numéro], [raison]. C’est bien ça ? »\n"
            f"  - Tu attends la réponse de l’appelant.\n"
            f"  - SEULEMENT APRÈS avoir entendu « oui c’est ça » ou équivalent → tu appelles take_message.\n"
            f"  - Ensuite tu dis la phrase finale.\n"
            f"  - Puis tu appelles end_call.\n"
            f"- Si l’appelant ne confirme pas ou corrige → tu ajustes sans appeler le tool.\n"
            f"- IMMÉDIATEMENT après avoir appelé take_message (sans attendre quoi que ce soit d'autre), dis EXACTEMENT et SANS DÉLAI cette phrase finale : « Parfait, je transmets votre message dès que possible. Merci d'avoir appelé ! Passez une belle journée ! »\n"
            f"- Dès que tu appelles take_message, dis IMMÉDIATEMENT quelque chose de naturel comme « Une petite seconde... Parfait, je transmets votre message dès que possible. Merci d'avoir appelé ! Passez une belle journée ! » pour combler tout délai éventuel.\n"
            f"- Ne laisse JAMAIS de silence prolongé après l'appel du tool. Parle tout de suite, même si le SMS est encore en cours d'envoi.\n"
            f"- Cette phrase doit être ta DERNIÈRE réponse verbale avant d'appeler end_call.\n"                
            f"Demande d'informations générales (heures, adresse, etc.) :\n"
            f"- Réponds brièvement et poliment à la question (ex. : heures d'ouverture, adresse).\n"
            f"- Ensuite, demande naturellement : « Est-ce que je peux vous aider avec autre chose ? » ou « Y'a-tu autre chose que je peux faire pour vous ? »\n"
            f"- Si l'appelant dit non, merci, au revoir, ou reste silencieux (5-10 secondes),\n"
            f"- Conclus immédiatement avec « Parfait ! Passez une belle journée ! » ou « Merci d'avoir appelé, bonne journée ! »\n"
            f"- Puis appelle IMMÉDIATEMENT le tool end_call.\n"
            f"- Ne relance pas plusieurs fois. Ne pose plus de questions.\n"
            f"Les bureau son ouvert du {company_hours}.\n"
            f"L'adresse est le {company_address}. \n"
            f"Fin d'appel : termine toujours poliment avec « Bonne journée ! » ou « Passez une belle journée ! » selon le contexte.\n"
            f"Reste toujours dans ton rôle de réceptionniste. Ne mentionne jamais que tu es une IA ou que tu as des limitations techniques, sauf si on te le demande explicitement. Si l'appelant est impoli ou hors sujet, reste calme et professionnelle et propose de transférer ou de prendre un message.\n"
            f"Quand tu dois répéter, confirmer ou dicter un numéro de téléphone, fais-le TRÈS lentement et TRÈS clairement. \n"
            f"Prononce les chiffres un par un (ou par petits groupes de 2-3 maximum) avec des pauses naturelles entre chaque groupe. \n"
            f"Exemple pour le numéro (514) 947-4976 :\n"
            f"« Cinq... un... quatre... neuf... quatre... sept... quatre... neuf... sept... six. »\n"
            f"Ou de façon plus naturelle au Québec : « Cinq un quatre... neuf quatre sept... quatre neuf sept six. »\n"
            #f"Insiste sur les pauses et parle posément pour que l'appelant puisse noter facilement. \n"
            #f"Répète toujours le numéro complet au moins une fois pour confirmation.\n"
            f"Fin d'appel générale (pour tous les cas sans prise de message ou quand la demande est satisfaite) :\n"
            f"- Quand l'appelant a eu sa réponse et dit qu'il n'a besoin de rien d'autre (ou reste silencieux 10-15 secondes après ta question « Autre chose ? »),\n"
            f"- Dis poliment « Parfait, merci d'avoir appelé ! Passez une belle journée ! »\n"
            f"- Puis appelle IMMÉDIATEMENT le tool end_call.\n"
            f"- Si silence prolongé à tout moment (plus de 20 secondes sans réponse), applique la même clôture sans relance supplémentaire.\n"        
            )

        if caller_number:
            base_instructions += (
                f"Information importante : l'appelant utilise actuellement le numéro de téléphone {caller_number}. Proposez-lui d'utiliser ce numéro (en le confirmant avec lui) s'il désire être rappelé ou s'il souhaite laisser un message.\n"
            )

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
                end_call,
                take_message
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
async def end_call(ctx: RunContext):
    """Termine l'appel en cours en supprimant la room. À appeler après avoir dit au revoir."""
    logger.info("Tool end_call appelé – fin de conversation imminente")
    # Attend que l'agent ait fini de parler entièrement
    await ctx.wait_for_playout()
    
    # Pause naturelle pour éviter toute coupure
    await asyncio.sleep(5.0)
    
    job_ctx = get_job_context()
    if not job_ctx:
        logger.warning("Impossible de récupérer le job context dans end_call")
        return None  # Rien à dire, évite de générer du text supplémentaire
    
    room_name = job_ctx.room.name
    logger.info(f"Suppression de la room {room_name} pour terminer l'appel proprement")
    
    try:
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(room=room_name)
        )
        logger.info("Room supprimée avec succès → SIP BYE envoyé")
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la room : {e}")
    
    return None  # Important : retourne None pour ne rien ajouter à la conversation (évite double au revoir)           

@function_tool
async def take_message(ctx: RunContext, name: str, callback_number: Optional[str] = None, reason: str = ""):
    """Enregistre un message laissé par l'appelant et envoie un SMS à l'équipe Telnek."""
    await ctx.wait_for_playout()  # Au cas où, pour ne pas couper Amélie
    
    job_ctx = get_job_context()
    if not job_ctx:
        logger.warning("Job context indisponible dans take_message")
        return None
    
    room = job_ctx.room
    
    # Récupère le numéro appelant réel (via participant SIP)
    sip_participant = next(
        (p for p in room.remote_participants.values() 
         if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP),
        None,
    )
    caller_number = "inconnu"
    if sip_participant and sip_participant.identity.startswith("sip_"):
        caller_number = sip_participant.identity[4:]  # enlève "sip_"
        # Optionnel : nettoyer +1 si présent
        if caller_number.startswith("+1"):
            caller_number = caller_number[2:]
    
    # Si pas de numéro de rappel spécifié → utilise le numéro appelant
    final_callback = callback_number or caller_number
    
    # Envoie le SMS via Twilio
    try:
        client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
        body = (
            f"📩 Nouveau message Telnek !\n\n"
            f"👤 De : {name}\n"
            f"📞 Appelant : {format_phone(caller_number)}\n"
            f"🔄 Rappel au : {format_phone(final_callback)}\n"
            f"💬 Message : {reason}\n\n"
            f"Heure : {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )        
        message = client.messages.create(
            to=admin_phone,
            from_=callee_number,
            body=body
        )
        logger.info(f"SMS envoyé avec succès (SID: {message.sid}) pour {name}")

        # === NOUVEAU : SMS de confirmation à l'appelant (pour tester) ===
        confirmation_body = (
            "Merci ! 😊\n"
            "Votre message a bien été transmis à l'équipe Telnek.\n"
            f"Nous vous rappelons au {format_phone(final_callback)} dès que possible.\n"
            "Passez une belle journée !\n"
            "Amélie, réceptionniste virtuelle Telnek"
        )
        confirmation_message = client.messages.create(
            to=final_callback,  # Ou caller_number si tu préfères forcer le numéro appelant
            from_=callee_number,
            body=confirmation_body
        )
        logger.info(f"SMS confirmation envoyé à l'appelant (SID: {confirmation_message.sid}) – {final_callback}")

    except Exception as e:
        logger.error(f"Erreur envoi SMS Twilio : {e}")
        
    return None  # Le modèle ne dira rien automatiquement du tool

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
#def prewarm(proc: JobProcess):
#    proc.userdata["vad"] = silero.VAD.load()

#async def entrypoint(ctx: JobContext):

    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    #session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        #stt=inference.STT(model="deepgram/nova-3", language="multi"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        #llm=inference.LLM(model="openai/gpt-4.1-mini"),
    #    llm=xai.realtime.RealtimeModel(
    #        voice="ara",                # default voice; "ara", others listed
            # Optional: custom turn detection (server VAD is used by default)
            # turn_detection=None,            # to disable built-in turn detection
            # or customize:
            # turn_detection=turn_detection.ServerVad(
            #     threshold=0.5,
            #     silence_duration_ms=250,
            #     prefix_padding_ms=300,
            # ),
    #    ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        #tts=inference.TTS(
        #    model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        #),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        #turn_detection=MultilingualModel(),
    #    vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
    #    preemptive_generation=True,
    #)

    session = AgentSession(
        stt=deepgram.STT(
            language="fr-CA",       # Accent québécois bien géré
            interim_results=True,   # Transcripts en temps réel
        ),
        llm=xai.realtime.RealtimeModel(
            voice="ara",
        ),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Détection du client par le nom de la room
    logger.info(f"Room name: {ctx.room.name}")
    if ctx.room.name.startswith("telnek-"):
        room_prefix = "telnek-"
        company_name = "Telnek"
        company_address = "sept cents soixante et quatre, Avenue Prieur à Laval, Québec. H7E 2V3"
        company_hours = "lundi au vendredi de 9 heure du matin a 5 heure de l'après-midi"
        admin_phone = "+15149474976"
        callee_number = "+14388147547"    
    elif ctx.room.name.startswith("bell-"):
        room_prefix = "bell-"
        company_name = "Bell"
        company_address = "CP 8787, succursale Centre-ville, Montréal, QC H3C 4R5"
        company_hours = "lundi au vendredi de 9 heure du matin a 5 heure de l'après-midi"
        admin_phone = "+15149474976"
        callee_number = "+14388141491"
    else:
        room_prefix = "Inconnue"
        company_name = "Inconnue"
        company_address = "Inconnue"
        company_hours = "Inconnue"
        admin_phone = "Inconnue"
        callee_number = "Inconnue"

    globals()["admin_phone"] = admin_phone
    globals()["callee_number"] = callee_number

    logger.info(f"company_name: {company_name}")
    logger.info(f"company_address: {company_address}")
    logger.info(f"company_hours: {company_hours}")
    logger.info(f"admin_phone: {admin_phone}")
    logger.info(f"callee_number: {callee_number}")


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
        if len(parts) >= 3 and parts[0] == room_prefix:  # ou "appel-" si pas de tiret supplémentaire
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
    assistant = Assistant(
        caller_number=caller_number,
        company_name=company_name,
        company_address=company_address,
        company_hours=company_hours,
        admin_phone=admin_phone
        )

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

    # Logging des transcripts en temps réel (client et Amélie)
    def on_transcription(transcription: rtc.Transcription):
        if transcription.segments:
            text = " ".join(seg.text for seg in transcription.segments).strip()
            if not text:
                return
        
            participant = transcription.participant
            if participant and participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                logger.info(f"👤 Client a dit : {text}")
            else:
                logger.info(f"🤖 Amélie a dit : {text}")

    # Abonne à l'événement de transcription de la room
    ctx.room.on("transcription_received", on_transcription)
    logger.info("Logging des transcripts activé via room events (client et Amélie)")

    # greeting immédiat pour les appels entrants (Twilio/SIP)

    # Greeting fixe et fiable via le modèle realtime
    welcome_message = f"Bonjour, vous êtes bien chez {company_name}, mon nom est {agent_name}. Comment puis-je vous aider aujourd’hui ?"

    greeting_instructions = (
        f"Dis EXACTEMENT ceci comme première phrase, sans rien ajouter, sans rien modifier et sans poser d'autre question : "
        f"« {welcome_message} » "
        f"Parle calmement, chaleureusement et avec un sourire naturel."
    )

    logger.info(f"Message de bienvenue forcé : {welcome_message}")

    await session.generate_reply(
        instructions=greeting_instructions,
        allow_interruptions=True  # L'appelant peut couper le greeting s'il parle tout de suite
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
#    worker = Worker(
#        entrypoint_fnc=entrypoint,          # ← note le "_fnc"
#        options=WorkerOptions(
#            concurrency=5,                  # ← commence par 5, monte à 10-15 après tests
#        ),
#        setup_fnc=prewarm,
#    )
#    agents.run("amelie", worker)            #
