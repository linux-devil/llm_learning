from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager
from langmem import create_memory_manager
from pydantic import BaseModel, Field


class Episode(BaseModel):  # 
    """Write the episode from the perspective of the agent within it. Use the benefit of hindsight to record the memory, saving the agent's key internal thought process so it can learn over time."""

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
        ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )


#  The Episode schema becomes part of the memory manager's prompt,
# helping it extract complete reasoning chains that guide future responses
manager = create_memory_manager(
    "openai:gpt-4o-mini",
    schemas=[Episode],
    instructions="Extract examples of successful explanations, capturing the full chain of reasoning. Be concise in your explanations and precise in the logic of your reasoning.",
    enable_inserts=True,
)


# Set up vector store for similarity search
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Configure memory manager with storage
manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("memories", "episodes"),
    schemas=[Episode],
    instructions="Extract exceptional examples of noteworthy problem-solving scenarios, including what made them effective.",
    enable_inserts=True,
)

llm = init_chat_model("openai:gpt-4o-mini")


@entrypoint(store=store)
def app(messages: list):
    # Step 1: Find similar past episodes
    similar = store.search(
        ("memories", "episodes"),
        query=messages[-1]["content"],
        limit=1,
    )

    # Step 2: Build system message with relevant experience
    system_message = "You are a helpful assistant."
    if similar:
        system_message += "\n\n### EPISODIC MEMORY:"
        for i, item in enumerate(similar, start=1):
            episode = item.value["content"]
            system_message += f"""

            Episode {i}:
            When: {episode['observation']}
            Thought: {episode['thoughts']}
            Did: {episode['action']}
            Result: {episode['result']}
            """

    # Step 3: Generate response using past experience
    response = llm.invoke([{"role": "system", "content": system_message}, *messages])

    # Step 4: Store this interaction if successful
    manager.invoke({"messages": messages})
    return response


app.invoke(
    [
        {
            "role": "user",
            "content": "What's a binary tree? I work with family trees if that helps",
        },
    ],
)
print(store.search(("memories", "episodes"), query="Trees"))