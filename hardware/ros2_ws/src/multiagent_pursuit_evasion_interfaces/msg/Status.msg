uint8 MAPE_STATUS_OFF = 0
uint8 MAPE_STATUS_TAKEOFF = 1
uint8 MAPE_STATUS_INITIALIZED = 2
uint8 MAPE_STATUS_RUNNING = 3
uint8 MAPE_STATUS_BLUE_WON = 4
uint8 MAPE_STATUS_RED_WON = 5

uint8 status
bool solver_ready

uint8 n_evaders
uint8 n_pursuers

string[] evader_cf_names
string[] pursuer_cf_names
uint8[] pursuer_targets

multiagent_pursuit_evasion_interfaces/EvaderState[] cf_evader_states
multiagent_pursuit_evasion_interfaces/PursuerState[] cf_pursuer_states