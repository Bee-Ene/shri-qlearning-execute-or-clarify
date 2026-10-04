# shri-qlearning-execute-or-clarify
Q-LEARNING FOR EXECUTE-OR-CLARIFY DECISIONS IN SYMBIOTIC HUMAN-ROBOT INTERACTION    
Key results: 100% policy accuracy, 92% user study success rate, r=−0.530 correlation
System requirements: ROS2 Humble, Python 3.11, Gazebo Classic 11

Includes: 

Python ROS2 package files : shri_state.py, shri_decision_node.py, combined_input_node.py, session_logger_node.py, gtts_tts_node.py, command_input_node.py, feedback_input_node.py, whisper_stt_node.py. 

Colab training scripts: QL.ipynb, Comparisons.ipynb, BaselineComp.ipynb, SensitivityAnalysis.ipynb, UserS.ipynb.
Key output figures
q_tables (3 .npy files)
Gazebo world files
UnitreeGo2 mesh files
SDF models
 

QL.ipynb (first), SensitivityAnalysis.ipynb, Comparisons.ipynb, BaselineComp.ipynb, ROS2 package files, UserS (last).

PS: command_input and combined_input serve the same purpose but in different situations, while, session_logger was used during user study.
