Implementing Long-Term Memory in AI Agents (Semantic, Episodic, Procedural) with LangMem

AI agents powered by large language models (LLMs) can appear more intelligent and personalized when they remember information over time. By equipping agents with long-term memory, developers enable them to recall facts, past interactions, and learned skills beyond a single chat session. In this article, we’ll conduct a deep dive into the role of memory in AI agents, focusing on the three key types of memory – semantic, episodic, and procedural – and how to implement each.

We’ll explore conceptual differences between these memory types, practical strategies for integrating memory into AI systems, and specific techniques using the LangMem framework (a toolkit for long-term memory in LangChain). We’ll also discuss optimization techniques like delayed memory processing, dynamic namespaces, efficient retrieval, and the performance trade-offs involved in giving your AI a long-term memory.

Conceptual Understanding: Memory Types in AI Agents

Memory Type	Purpose	Agent Example	Human Example	Typical Storage Pattern
Semantic	Facts & Knowledge	User preferences; knowledge triplets	Knowing Python is a programming language	Profile or Collection
Episodic	Past Experiences	Few-shot examples; Summaries of past conversations	Remembering your first day at work	Collection
Procedural	System Behavior	Core personality and response patterns	Knowing how to ride a bicycle	Prompt rules or Collection
Source : https://blog.langchain.dev/langmem-sdk-launch/

Just as humans have multiple forms of memory, AI agents can benefit from different memory types for different purposes . In cognitive terms, we can draw analogies to human memory when designing AI agent memory:

• Semantic Memory (Facts & Knowledge): Semantic memory is about storing factual information or general knowledge an agent has learned . In humans, this is like remembering that Paris is the capital of France or Python is a programming language. For AI, semantic memory might include facts about the user or world that were learned during interactions or provided as data (e.g. a user’s name, preferences, key domain facts) . This memory enables the agent to ground its responses with correct details and personalization. Example: A virtual assistant’s semantic memory could store that the user’s favorite cuisine is Italian and their birthday is July 20th, so it can later recommend Italian restaurants or send birthday wishes.

• Episodic Memory (Events & Experiences): Episodic memory records specific experiences or past events . For humans, episodic memories might be recollections of your first day at work or a memorable trip. In AI agents, episodic memory means remembering past dialogues or problem-solving episodes – essentially the agent’s own experiences in dealing with certain situations . This could include summaries of previous conversations, successful task outcomes, or mistakes made, along with the context in which they occurred.

Episodic memory helps the agent recall “how did I handle this before?” and apply that experience to guide future behavior . Example: A customer support chatbot’s episodic memory might include a summary of the last support session with a user, so if the user returns, the bot remembers what was tried before and what the result was.

• Procedural Memory (Skills & Behaviors): Procedural memory captures the know-how for performing tasks, encompassing rules, skills, or policies the agent follows . In humans, this is like the ingrained skill of riding a bicycle or playing the piano – you might not recall a specific event, but you have internalized how to do it. For AI agents, procedural memory manifests in the agent’s core behavior: it can be encoded in the model’s weights, in the agent’s code, or importantly in the system prompts and instructions that guide its responses . By updating its procedural memory, an agent can learn new behaviors or refine its style over time without changing its underlying model weights . Example: An AI coding assistant might learn over time to adopt a more detailed code commenting style after it consistently gets user feedback asking for more explanation. This learned behavior is stored as an adjustment to its system prompt (procedural memory), so future code outputs include better comments by default.

Semantic memory gives the agent a knowledge base of facts (the “**what”)
Episodic memory provides it with personal experiences (the “when and how” of past events)
Procedural memory governs its inherent skills or behaviors (the “how to do” rules).

In practice, an AI agent will typically use a combination of all three to achieve more intelligent and personalized interactions. For example, a sophisticated personal assistant might use semantic memory to recall a user’s preferences, episodic memory to remember the context of previous conversations with that user, and procedural memory to adapt its tone or strategy based on what has been effective in the past.
