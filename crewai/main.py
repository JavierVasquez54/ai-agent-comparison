import os
from dotenv import load_dotenv
from crewai import Crew, Process

# Handle both relative and absolute imports
try:
    from .agents import research_analyst, investment_stratergist, report_writer
    from .tasks import research_task, analysis_task, report_task
except ImportError:
    from agents import research_analyst, investment_stratergist, report_writer
    from tasks import research_task, analysis_task, report_task

load_dotenv()


# --- Main execution block ---
if __name__ == "__main__":
    # Get user input for the stock ticker
    stock_ticker = input("Enter the stock ticker you want to analyze (e.g. NVDA, AMD): ").upper()

    # Create the crew with the defined agents and tasks
    stock_crew = Crew(
        name="Stock Analysis Crew",
        agents=[research_analyst, investment_stratergist, report_writer],
        tasks=[research_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True,
        max_rpm=5000
    )

    # Use the kikoff method with a dynamic input
    # The {ticket} placeholder in the tasks will be replaced with the user's input.
    result = stock_crew.kickoff(inputs={"ticker": stock_ticker})

    print("\n\n--- FINAL REPORT (CrewAI with Groq) ---")
    print(result)