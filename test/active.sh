#!/bin/zsh

# Check if the correct number of arguments is given
if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <pcap_file> <flow_tuple> <ACTIVE_TIMEOUT>"
    exit 1
fi

PCAP_FILE=$1
FLOW_TUPLE=$2
ACTIVE_TIMEOUT=$3
CLUMP_TIMEOUT=$4

IFS=',' read -r src_ip dst_ip src_port dst_port protocol <<< "$FLOW_TUPLE"

# Use tshark to read the pcap file and filter the flow of interest
tshark -r "$PCAP_FILE" \
     -Y "((ip.src==$src_ip and tcp.srcport==$src_port and ip.dst==$dst_ip and tcp.dstport==$dst_port) or \
          (ip.src==$dst_ip and tcp.srcport==$dst_port and ip.dst==$src_ip and tcp.dstport==$src_port))" \
     -T fields -e frame.time_epoch | awk -v ACTIVE_TIMEOUT="$ACTIVE_TIMEOUT" -v CLUMP_TIMEOUT="$CLUMP_TIMEOUT" '{
  if (NF == 0) {
  }
  if (NR==1) {
      last_time = $1
      last_active = 0
      clump_time = $1
      start_time = $1
      #printf "%-20s %-11s %-13s %-13s\n", "time", "IAT", "Last Active", "Idle_Time"
      #printf "%20.6f %10.6f %12.6f %12.6f\n", $1, 0, last_active, 0
  } else {
      iat = $1 - last_time
      if (iat > CLUMP_TIMEOUT) {
        if (iat - last_active > ACTIVE_TIMEOUT){
          #printf "%20.6f %10.6f %12.6f %12.6f %-3s\n", $1, iat, last_active, iat - last_active, "*"
          value = iat - last_active
          printf "%16.6f\n", value
        }else{
          #printf "%20.6f %10.6f %12.6f %12.6f %-3s\n", $1, iat, last_active, 0, "-"
        }
        last_active = iat
      } else{
          #printf "%20.6f %10.6f %12.6f %12.6f %-3s\n", $1, iat, last_active, 0, "+"
      }
      last_time = $1
  }
}'
