from agent.agent_KT import KTAgent
def get_agent(config):
    if config.module == 'KT':
        return KTAgent(config)
    else:
        raise ValueError
