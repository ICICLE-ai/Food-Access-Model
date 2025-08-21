# parallel_scheduler.py
import concurrent.futures
import logging
import threading
from functools import partial
from mesa.time import BaseScheduler


def run_agent_step(agent):
    # thread_id = threading.get_ident()
    # print(f"[AGENT {agent.unique_id}] Step called in thread {thread_id}", flush=True)
    return agent.step()



class ParallelScheduler(BaseScheduler):
    """
    A custom Mesa scheduler that executes each agent's step() in parallel
    using multiple threads.
    """

    def step(self):
        # Get a list of agents (households) at the start of this step
        agent_list = list(self._agents.values())
        # agent_list = [raw_agent_list[0], raw_agent_list[1]]
        # print("# AGENTS TO BE PROCESSED", len(agent_list), flush=True)
        

        # with open("output.log", "a") as file:
        #     print(f"Scheduler launching {len(agent_list)} agent steps in parallel", file=file)

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            # Submit each agent's step to a separate thread
            futures = {
                executor.submit(run_agent_step, agent): agent
                for agent in agent_list
            }
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                agent = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Agent {agent.unique_id} step failed: {e}")

            self.steps += 1
            self.time += 1
