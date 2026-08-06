import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import llm, stt, tts, inference
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
import time

logger = logging.getLogger(__name__)

load_dotenv()

from livekit.agents.llm.mcp import MCPServerHTTP

# Define your agent's behavior by extending the Agent class
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a professional voice AI assistant for First Apex National Bank.
Help the caller with their loan inquiries, product options, policy questions, and payment calculations.
Always keep spoken replies concise (1 to 3 sentences max), conversational, and friendly for voice.
When a customer asks for loan payments or eligibility, use your MCP tools to check official bank policies and your calculation tool to compute the exact monthly payment."""
        )

    # Native LiveKit Function Tool for calculating loan EMI
    @llm.function_tool
    async def calculate_loan_emi(
        self, principal: float, interest_rate: float, tenure_years: int
    ) -> str:
        """
        Calculates the exact monthly loan repayment (EMI), total interest, and total cost of the loan.

        Args:
            principal: Total loan amount borrowed in dollars (e.g. 250000).
            interest_rate: Annual interest rate percentage (e.g. 6.5 for 6.5%).
            tenure_years: Loan duration in years (e.g. 15, 20, 30).
        """
        if principal <= 0 or interest_rate <= 0 or tenure_years <= 0:
            return "Please provide positive numbers for principal, interest rate, and tenure."

        monthly_rate = (interest_rate / 100) / 12
        total_months = tenure_years * 12
        emi = (principal * monthly_rate * ((1 + monthly_rate) ** total_months)) / (
            ((1 + monthly_rate) ** total_months) - 1
        )
        total_payment = emi * total_months
        total_interest = total_payment - principal

        return (
            f"Loan Calculation Summary:\n"
            f"- Principal: ${principal:,.2f}\n"
            f"- Interest Rate: {interest_rate}%\n"
            f"- Tenure: {tenure_years} years ({total_months} months)\n"
            f"- Estimated Monthly Payment: ${emi:,.2f}\n"
            f"- Total Interest Paid: ${total_interest:,.2f}\n"
            f"- Total Loan Cost: ${total_payment:,.2f}"
        )

server = AgentServer()

# The entrypoint function runs when a participant joins the room
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Configure the voice pipeline with STT, LLM, TTS, VAD, and MCP providers
    session = AgentSession(
        # LLM with fallback: OpenAI primary, Gemini backup
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash"),
            ]
        ),
        # STT with fallback: AssemblyAI primary, Deepgram backup
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("assemblyai/universal-streaming:en"),
                inference.STT.from_model_string("deepgram/nova-3"),
            ]
        ),
        # TTS with fallback: Cartesia primary, Inworld backup
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string("cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),  # Semantic turn detection
        preemptive_generation=True,
        # MCP Server connections for tools and external knowledge
        mcp_servers=[
            MCPServerHTTP(
                url="http://127.0.0.1:8000/sse",
                transport_type="sse",
            )
        ],
    )

    #########################################
    #---- START OF METRICS COLLECTION -----#
    #########################################

    # Aggregate data across all conversation turns
    usage_collector = metrics.UsageCollector()

    # Track End of Utterance timing (when turn detector decides user finished speaking)
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        # Capture EOU metrics for TTFA calculation
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        # Log each metric as it arrives and add to usage collector
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)


    async def log_usage():
        # Print per-session summary (tokens, audio duration, costs)
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    # Fire log_usage when worker shuts down
    ctx.add_shutdown_callback(log_usage)

    # Track end of utterance to first audio (TTFA)
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            if last_eou_metrics:
                # Calculate time since user finished speaking
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"Time to first audio: {elapsed:.3f}s")

    #########################################
    #---- END OF METRICS COLLECTION -----#
    #########################################

    # Start the session with noise cancellation enabled
    await session.start(        
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server) 