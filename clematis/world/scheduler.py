from __future__ import annotations


class RoundRobinScheduler:
    def __init__(self, agents):
        self.agents = agents
        self.i = 0

    def choose_agent(self, world_state):
        a = self.agents[self.i % len(self.agents)]
        self.i += 1
        return a

    def on_timeout(self, agent_id):
        pass
