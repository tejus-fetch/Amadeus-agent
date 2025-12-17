from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    from app.agent import create_agent

    agent = create_agent()
    agent.run()
