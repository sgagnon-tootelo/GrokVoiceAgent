import logging
import asyncio
import os

from twilio.rest import Client
from typing import Optional
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup

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
#from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import deepgram
#from livekit.agents import Worker, WorkerOptions

from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

# Fuseau horaire du Québec
TZ_MONTREAL = ZoneInfo("America/Montreal")

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

def spoken_phone(raw_number: str) -> str:
    """
    Convertit un numéro de téléphone en version phonétique québécoise lente,
    groupe par 3-3-4, avec tous les chiffres prononcés séparément.
    Exemple : "4508080813" → "quatre cinq zéro... huit zéro huit... zéro huit treize"
    """
    # Nettoie : ne garde que les chiffres, enlève +1 si présent
    number = ''.join(filter(str.isdigit, raw_number))
    if number.startswith('1') and len(number) == 11:
        number = number[1:]
    
    if len(number) != 10:
        return "numéro inconnu"  # fallback safe
    
    digits_map = {
        '0': 'zéro',
        '1': 'un',
        '2': 'deux',
        '3': 'trois',
        '4': 'quatre',
        '5': 'cinq',
        '6': 'six',
        '7': 'sept',
        '8': 'huit',   # le "t" est dans l'orthographe normale → la voix ara le prononce généralement bien
        '9': 'neuf'
    }
    
    def speak_group(group: str) -> str:
        return ' '.join(digits_map[d] for d in group)
    
    area_code = speak_group(number[:3])
    exchange = speak_group(number[3:6])
    subscriber = speak_group(number[6:10])  # 4 chiffres, tous séparés
    
    # Les "..." indiquent des pauses naturelles (le modèle les respecte bien)
    return f"{area_code}... {exchange}... {subscriber}"

# Charge les vars personnalisées depuis le .env (avec fallback Telnek pour tes tests)
agent_name = os.getenv("AGENT_NAME", "Amélie")

class Assistant(Agent):
    def __init__(
            self, 
            caller_number: Optional[str] = None,
            formatted_caller: Optional[str] = None,
            spoken_caller: Optional[str] = None,
            company_name: str = "Telnek",
            company_address: str = "",
            company_hours: str = "",
            admin_phone: str = "",
            instructions_specific: str = ""
            ) -> None:
        self.formatted_caller = formatted_caller or "inconnue"
        self.spoken_caller = spoken_caller or "inconnue"
        self.room: rtc.Room | None = None
        self.admin_phone = admin_phone

        logger.debug(f"AGENT_NAME: {agent_name}")
        logger.debug(f"COMPANY_NAME: {company_name}")
        logger.debug(f"COMPANY_ADDRESS: {company_address}")
        logger.debug(f"COMPANY_HOURS: {company_hours}")
        logger.debug(f"admin_phone: {admin_phone}")
        logger.debug(f"instructions_specific: {instructions_specific}")

        base_instructions = (
            f"Tu es {agent_name}, une réceptionniste virtuelle TRÈS chaleureuse, professionnelle et efficace pour la compagnie {company_name}.\n"
            f"Imagine que tu souris largement en parlant — rends ta voix encore plus accueillante, sympathique et réconfortante.\n"
            f"Tu parles en français québécois courant et poli, avec un ton naturel comme une vraie personne au téléphone au Québec.\n"
            f"Tes réponses doivent être courtes et naturelles : maximum 2-3 phrases à la fois. Parle à un rythme détendu, avec des pauses naturelles.\n"
            f"Utilise des contractions courantes (« j’peux », « c’est », « y’a », « j’vas », « laissez-moi »), mais RESTE TOUJOURS POLIE ET PROFESSIONNELLE.\n"
            f"Évite ABSOLUMENT les expressions trop familières comme « bein », « chu », « moé », « toé ». Dis toujours « bien », « je suis », « moi », « vous ».\n"
            f"Toujours vouvoyer l’appelant : utilise « vous », « laissez-moi », « pourriez-vous », etc. Jamais de tutoiement.\n"
            f"Tu peux poursuivre en anglais si l’appelant est clairement anglophone.\n\n"

            f"CRUCIAL : Tu DOIS TOUJOURS poser UNE SEULE question ou demande à la fois. Jamais deux ou plus dans la même réponse.\n"
            f"Exemple à ÉVITER : « Quel est votre nom et quel est le sujet ? »\n"
            f"Exemple correct : Demande d’abord une chose, attends la réponse complète, puis passe à la suivante.\n"
            f"Progresse calmement, étape par étape, sans jamais regrouper ou anticiper.\n\n"

            f"Quand l'appel commence, salue comme ça : « Bonjour, vous êtes bien chez {company_name}, mon nom est {agent_name}. Comment puis-je vous aider aujourd’hui ? »\n\n"

            f"Prise de message ou rendez-vous :\n"
            f"- Commence par demander la personne recherchée ou le département.\n"
            f"- Ensuite, demande le sujet ou la raison de l'appel (une seule question).\n"
            f"- Propose d'abord d'utiliser le numéro actuel pour le rappel : « Je peux utiliser le numéro d'où vous appelez, qui est le [numéro formaté lentement], ou préférez-vous m'en donner un autre ? »\n"
            f"- Si l'appelant confirme le numéro actuel ou en donne un autre, note-le sans répéter inutilement.\n"
            f"- Demande le nom complet seulement quand c'est nécessaire, et toujours séparément.\n"
            f"- Une fois toutes les infos recueillies, répète UNE SEULE FOIS pour confirmation : « Juste pour confirmer : [nom], [numéro], [message/sujet]. C’est bien ça ? »\n"
            f"- Pose toujours UNE SEULE question ou demande à la fois. Attends la réponse complète de l’appelant avant de continuer. Progresse étape par étape, calmement.\n"
            f"- Une fois confirmé, appelle le tool take_message avec les paramètres exacts (name, callback_number, reason).\n"
            f"- CRUCIAL : Après avoir appelé le tool take_message, dis IMMÉDIATEMENT sans attendre le résultat cette phrase finale comme dernière réponse : « Parfait, je transmets votre message dès que possible. Merci d'avoir appelé ! Passez une belle journée ! Au revoir ! »\n"
            f"- IMMÉDIATEMENT après avoir fini de dire cette phrase (et seulement après), appelle le tool end_call pour terminer l'appel.\n"
            f"- Ne dis RIEN d'autre. Ne pose plus de question. Ne relance pas.\n"
            f"- CRUCIAL : Tu NE DOIS JAMAIS appeler le tool take_message avant d’avoir entendu et reçu une confirmation EXPLICITE de l’appelant APRÈS le récapitulatif (ex. « oui », « c’est correct », « parfait », « c’est ça »).\n"
            f"- Si tu n’as pas encore la confirmation, tu NE FAIS RIEN et tu ATTENDS silencieusement la réponse.\n"
            f"- Ne anticipe JAMAIS la confirmation. Même si tout semble complet, attends toujours la réponse verbale.\n"
            f"- Si l’appelant ne confirme pas ou corrige → tu ajustes sans appeler le tool.\n"
            f"- Ne laisse JAMAIS de silence prolongé après l'appel du tool. Parle tout de suite, même si le SMS est encore en cours d'envoi.\n"
  
            f"Demande d'informations générales (heures, adresse, service offert etc.) :\n"
            f"- Réponds brièvement et chaleureusement.\n"
            f"- Ensuite, demande : « Est-ce que je peux vous aider avec autre chose ? » ou « Y a-t-il autre chose que je peux faire pour vous ? »\n"
            f"- Si l'appelant dit non ou reste silencieux (5-10 secondes), conclus avec : « Merci d'avoir appelé ! Passez une belle journée ! Au revoir ! »\n"
            f"- Puis appelle IMMÉDIATEMENT end_call.\n\n"

            f"Quand tu dois dire un numéro de téléphone, utilise TOUJOURS ce format précis et prononce-le lentement, groupe par groupe, à la québécoise :\n"
            f"- Groupe par 3-3-4 : ex. pour (514) 947-4976 → « cinq un quatre... neuf quatre sept... quatre neuf sept six. »\n"
            f"- Exemple pour (450) 808-0813 : « quatre cinq zéro... huit zéro huit... zéro huit un trois. »\n"
            f"- Prononce toujours le 't' final de 'huit' clairement : « huit » (pas « hui »).\n"
            f"- Pour 'quatre' dis « quatre », pour 'cinq' dis « cinq », etc. Jamais de style européen comme « quatre-vingt ».\n"
            f"- Pause naturelle d’environ 1 seconde entre chaque groupe de chiffres.\n"
            f"- Ne dis jamais le numéro en continu ou en format international (+1…). Utilise toujours ce format phonétique québécois.\n"
 
            f"- Pause naturelle entre chaque groupe.\n\n"

            f"Nos bureaux sont ouverts du {company_hours}.\n"
            f"L'adresse de nos bureau est le {company_address}.\n"
            f"Reste toujours dans ton rôle. Ne mentionne jamais que tu es une IA.\n"
            f"Si silence prolongé (>20 secondes), conclus poliment et appelle end_call.\n\n"

            f"Quand l'appelant demande l'heure, la date ou le jour, utilise IMMÉDIATEMENT la tool get_current_datetime ou get_current_time.\n"
            f"Pour infos détaillées sur le site, utilise IMMÉDIATEMENT fetch_company_website.\n\n"

            f"RÉSUMÉ DES RÈGLES ABSOLUES :\n"
            f"- Une seule question à la fois.\n"
            f"- Attendre confirmation EXPLICITE avant tout tool.\n"
            f"- Phrase finale TOUJOURS complète, immédiate et chaleureuse.\n"
            f"- Ton ultra-chaleureux, québécois poli (jamais trop familier).\n"
        )

        # numéro de l'appelant est connue
        if caller_number:
            base_instructions += (
                f"Information importante : l'appelant utilise actuellement le numéro de téléphone {caller_number}.\n"
                f"Propose d'abord d'utiliser le numéro actuel pour le rappel avec CETTE phrase EXACTE :\n"
                f"« Je peux utiliser le numéro d'où vous appelez, qui est le {self.spoken_caller}, ou préférez-vous m'en donner un autre ? »\n"
                f"Prononce lentement, avec des pauses naturelles après chaque groupe de chiffres.\n\n"
            )

        # ajour des instructions spécific pour cette compagnie
        if instructions_specific:
            base_instructions += instructions_specific

        # ajour pour tool fetch_company_website
        base_instructions += (
            f"Quand l'appelant demande des informations détaillées qui pourraient être sur le site web de {company_name} \n"
            f"(services, tarifs, équipe, coordonnées complètes, promotions, etc.), utilise IMMÉDIATEMENT le tool fetch_company_website \n"
            f"avec la section la plus pertinente ('accueil', 'services', 'contact', 'apropos', 'equipe'). \n"
            f"Si tu cherches quelque chose de précis, passe-le dans 'query'. \n"
            f"Ne l'utilise QUE pour l'entreprise en cours ({company_name}). \n"
            f"Ensuite, résume les infos de façon naturelle, concise et chaleureuse à l'appelant.\n"
        )

        # ajour pour tool get_current_time et get_current_datetime
        #base_instructions += (
        #    f"Quand l'appelant demande l'heure actuelle, utilise IMMÉDIATEMENT la tool get_current_time pour obtenir l'heure exacte à Montréal et réponds poliment avec cette information.\n"
        #    f"Quand l'appelant demande la date, le jour de la semaine ou l'heure, utilise IMMÉDIATEMENT la tool get_current_datetime (ou get_current_time pour l'heure seule) pour répondre précisément.\n"
        #)

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
                take_message,
                fetch_company_website,
                get_current_time,
                get_current_datetime,
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
    await asyncio.sleep(3.0)
    
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
    room_name = job_ctx.room.name

    # Détection de l'entreprise
    if room_name.startswith("telnek-"):
        company = "Telnek"
    elif room_name.startswith("electrizone-"):
        company = "ÉlectriZone"
    else:
        company = "Inconnue"

    
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
            f"📩 Nouveau message {company} !\n\n"
            f"👤 De : {name}\n"
            f"📞 Appelant : {format_phone(caller_number)}\n"
            f"🔄 Rappel au : {format_phone(final_callback)}\n"
            f"💬 Message : {reason}\n\n"
            f"Heure : {datetime.now(TZ_MONTREAL).strftime('%Y-%m-%d %H:%M')}"
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
            f"Votre message a bien été transmis à l'équipe {company}.\n"
            f"Nous vous rappelons au {format_phone(final_callback)} dès que possible.\n"
            "Passez une belle journée !\n"
            f"Amélie, réceptionniste virtuelle {company}"
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

@function_tool
async def fetch_company_website(ctx: RunContext, section: str = "accueil", query: str = "") -> str:
    """
    Récupère des informations actualisées directement du site web de l'entreprise en cours (Telnek ou ÉlectriZone).
    Utilise ce tool quand l'appelant demande des infos qui pourraient être sur le site (services, tarifs, équipe, etc.).

    Args:
        section: Section/page du site (ex: "accueil", "services", "contact", "apropos", "equipe").
                 Si inconnue, fallback sur l'accueil.
        query: Mot-clé ou question précise pour guider la recherche dans la page.
    """
    await ctx.wait_for_playout()

    job_ctx = get_job_context()
    if not job_ctx:
        return "Erreur interne : contexte indisponible."

    room_name = job_ctx.room.name

    # Détection de l'entreprise (même logique que dans my_agent)
    if room_name.startswith("telnek-"):
        company = "Telnek"
        base_url = "https://telnek.com"
        url_map = {
            "accueil": "/",
            "services": "/",
            "contact": "/",
            "courriel": "/",
            "nom du président": "/"
            }
    elif room_name.startswith("electrizone-"):
        company = "ÉlectriZone"
        base_url = "https://www.facebook.com/Electrizone?locale=fr_CA"
        url_map = {
            "accueil": "/",                     # Page d'accueil uniquement pour l'instant
            "license RBQ:": "https://www.construction411.com/electricians/st-pascal/electrizone/"
        }
    else:
        return "Désolé, je n'ai pas accès au site web pour cette entreprise pour le moment."

    # Résolution de l'URL
    path = url_map.get(section.lower().strip(), "/")
    url = base_url + path if path.startswith("/") else base_url + "/" + path

    logger.info(f"Tool fetch_company_website appelé → Entreprise: {company} | URL: {url} | Query: {query}")

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return f"Erreur : impossible de charger la page ({response.status}). Je peux vous donner les infos de base."

                html = await response.text()

                soup = BeautifulSoup(html, "html.parser")
                for element in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                    element.decompose()

                text = soup.get_text(separator="\n", strip=True)

                max_length = 12000
                if len(text) > max_length:
                    text = text[:max_length] + "\n\n... (texte tronqué)"

                result = f"Contenu de la section '{section}' du site {company} ({url}) :\n\n{text}"
                if query:
                    result += f"\n\nRecherche spécifique : {query}"

                return result

    except Exception as e:
        logger.error(f"Erreur fetch site {company} : {e}")
        return "Désolé, je n'arrive pas à accéder au site pour le moment. Je peux répondre avec les informations générales que je connais."

@function_tool
async def get_current_time(ctx: RunContext) -> str:
    """Retourne l'heure actuelle à Montréal (Québec). 
    Utilise cette tool quand l'appelant demande l'heure actuelle ou « quelle heure il est ? »."""
    
    await ctx.wait_for_playout()  # Optionnel mais recommandé : attend que l'agent ait fini de parler avant d'exécuter
    
    now = datetime.now(TZ_MONTREAL)
    heure = now.strftime("%H:%M")          # Format 14:30
    heure_parlee = now.strftime("%H heure %M")  # Pour prononciation naturelle
    
    # Retourner une phrase naturelle que le LLM pourra utiliser directement
    return f"Il est actuellement {heure_parlee} à Montréal."

@function_tool
async def get_current_datetime(ctx: RunContext) -> str:
    """Retourne la date complète et l'heure actuelle à Montréal (Québec), avec le jour de la semaine en français.
    Utilise cette tool quand l'appelant demande la date, le jour de la semaine, ou « on est quel jour ? », « quelle date on est ? », etc."""
    
    await ctx.wait_for_playout()
    
    now = datetime.now(TZ_MONTREAL)
    
    # Jours de la semaine en français québécois
    jours_fr = {
        "Monday": "lundi",
        "Tuesday": "mardi",
        "Wednesday": "mercredi",
        "Thursday": "jeudi",
        "Friday": "vendredi",
        "Saturday": "samedi",
        "Sunday": "dimanche"
    }
    
    # Mois en français
    mois_fr = {
        1: "janvier", 2: "février", 3: "mars", 4: "avril",
        5: "mai", 6: "juin", 7: "juillet", 8: "août",
        9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"
    }
    
    jour_semaine = jours_fr[now.strftime("%A")]
    jour = now.day
    mois = mois_fr[now.month]
    annee = now.year
    heure = now.strftime("%H:%M")
    
    # Phrase naturelle et chaleureuse
    return f"Aujourd'hui, on est {jour_semaine} le {jour} {mois} {annee}, et il est {heure} à Montréal."

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
        instructions_specific = (
            f"Telnek est une entreprise spécialisée dans les services de centre d'appels, de télémarketing et de centre de contact.\n"
        )    
    elif ctx.room.name.startswith("electrizone-"):
        room_prefix = "electrizone-"
        company_name = "électri-zone"
        company_address = "deux milles dix, rue Alphonse, à Saint-Pascal, Québec. G0L 3Y0"
        company_hours = "lundi au vendredi de 8 heure à 17 heure"
        admin_phone = "+15149474976"
        callee_number = "+14388141491"
        instructions_specific = (
            "ÉlectriZone se spécialise dans les services électriques résidentiel, commercial et agricole.\n"
            "Propriétaire : Guillaume Boucher.\n"
            "Région desservie : Kamouraska et environs.\n"
            "Pour plus de détails ou projets en cours, mentionne que nous sommes actifs sur Facebook (ÉlectriZone).\n"
            "Ajout pour la prise de message pour électrizone: \n"
            "- Après avoir la raison de l’appel, demande toujours : « Est-ce que c’est pour une installation résidentielle, commerciale ou agricole ? »\n"
            "- Attends la réponse avant de continuer vers le numéro/nom/récap.\n"
        ) 
    else:
        room_prefix = "Inconnue"
        company_name = "Inconnue"
        company_address = "Inconnue"
        company_hours = "Inconnue"
        admin_phone = "Inconnue"
        callee_number = "Inconnue"
        instructions_specific = ""

    globals()["admin_phone"] = admin_phone
    globals()["callee_number"] = callee_number

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

    # === NORMALISATION ET FORMATAGE DU NUMÉRO === (ton bloc existant)
    if caller_number:
        original = caller_number
        
        if caller_number.startswith("+1") and len(caller_number) >= 11:
            caller_number = caller_number.lstrip("+1")
        
        clean_digits = ''.join(filter(str.isdigit, original))
        if clean_digits.startswith('1') and len(clean_digits) == 11:
            clean_digits = clean_digits[1:]
        
        # Format visuel pour SMS et logs
        formatted_caller = format_phone(clean_digits)
        
        # Version parlée phonétique (notre nouvelle fonction)
        spoken_caller = spoken_phone(clean_digits)
        
        logger.info(f"Numéro appelant → formaté: {formatted_caller} | parlé: {spoken_caller}")
    else:
        formatted_caller = "inconnu"
        spoken_caller = "inconnu"
        clean_digits = ""

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
        formatted_caller=formatted_caller,
        spoken_caller=spoken_caller,
        company_name=company_name,
        company_address=company_address,
        company_hours=company_hours,
        admin_phone=admin_phone,
        instructions_specific=instructions_specific
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

    # Join the room and connect to the user
    await ctx.connect()


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
