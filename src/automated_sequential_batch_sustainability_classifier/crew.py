import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FirecrawlScrapeWebsiteTool





@CrewBase
class AutomatedSequentialBatchSustainabilityClassifierCrew:
    """AutomatedSequentialBatchSustainabilityClassifier crew"""


    @agent
    def web_content_scraper(self) -> Agent:

        return Agent(
            config=self.agents_config["web_content_scraper"],


            tools=[FirecrawlScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,


            max_execution_time=None,
            llm=LLM(
                model="anthropic/claude-sonnet-4-20250514",
                temperature=0.7,
            ),

        )

    @agent
    def sustainability_marketing_classifier(self) -> Agent:

        return Agent(
            config=self.agents_config["sustainability_marketing_classifier"],


            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,


            max_execution_time=None,
            llm=LLM(
                model="anthropic/claude-sonnet-4-20250514",
                temperature=0.7,
            ),

        )

    @agent
    def onedrive_data_manager(self) -> Agent:

        return Agent(
            config=self.agents_config["onedrive_data_manager"],


            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,

            apps=[
                    "microsoft_onedrive/get_file_by_path",

                    "microsoft_onedrive/upload_file",

                    "microsoft_onedrive/list_files",

                    "microsoft_onedrive/search_files",

                    "microsoft_onedrive/download_file_by_path",

                    "microsoft_onedrive/get_recent_files",

                    "microsoft_onedrive/list_files_by_path",

                    "microsoft_excel/get_worksheets",

                    "microsoft_excel/get_range_data",

                    "microsoft_excel/get_workbooks",
                    ],


            max_execution_time=None,
            llm=LLM(
                model="anthropic/claude-sonnet-4-20250514",
                temperature=0.7,
            ),

        )

    @agent
    def batch_processing_coordinator(self) -> Agent:

        return Agent(
            config=self.agents_config["batch_processing_coordinator"],


            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,


            max_execution_time=None,
            llm=LLM(
                model="anthropic/claude-sonnet-4-20250514",
                temperature=0.7,
            ),

        )



    @task
    def read_urls_from_onedrive_spreadsheet(self) -> Task:
        return Task(
            config=self.tasks_config["read_urls_from_onedrive_spreadsheet"],
            markdown=False,


        )

    @task
    def coordinate_batch_processing(self) -> Task:
        return Task(
            config=self.tasks_config["coordinate_batch_processing"],
            markdown=False,


        )

    @task
    def scrape_all_website_content_in_batches(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_all_website_content_in_batches"],
            markdown=False,


        )

    @task
    def classify_all_urls_for_sustainability_marketing(self) -> Task:
        return Task(
            config=self.tasks_config["classify_all_urls_for_sustainability_marketing"],
            markdown=False,


        )

    @task
    def save_results_to_onedrive_csv(self) -> Task:
        return Task(
            config=self.tasks_config["save_results_to_onedrive_csv"],
            markdown=False,


        )


    @crew
    def crew(self) -> Crew:
        """Creates the AutomatedSequentialBatchSustainabilityClassifier crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            chat_llm=LLM(model="anthropic/claude-sonnet-4-20250514"),
        )
