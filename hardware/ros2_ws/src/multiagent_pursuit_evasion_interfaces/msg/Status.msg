uint8 MAPE_STATUS_OFF = 0
uint8 MAPE_STATUS_INITIALIZED = 1
uint8 MAPE_STATUS_RUNNING = 2

uint8 status

uint8 n_evaders
uint8 n_pursuers

string[] evader_cf_names
string[] pursuer_cf_names
uint8[] pursuer_targets

multiagent_pursuit_evasion_interfaces/State[] cf_evader_states
multiagent_pursuit_evasion_interfaces/State[] cf_pursuer_states