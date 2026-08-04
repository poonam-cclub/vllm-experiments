#ncu --metrics \
#smsp__cycles_elapsed.sum,\
#smsp__inst_executed.sum,\
#smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active,\
#smsp__average_warps_issue_stalled_lg_throttle_per_issue_active,\
#dram__bytes.sum, $1

ncu --clock-control none --section SpeedOfLight $1
