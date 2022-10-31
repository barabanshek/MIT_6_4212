#!/bin/bash

# Portlist for Drake to forward.
port_string=""
for port in 8080 7000 7001 7002 7003 7004 7005 7006 7007 7008 7009 7010
do
  port_string="$port_string -L $port:localhost:$port"
done

# Connect to server and exec python stuff.
ssh ${port_string} nikita@dnp-ext-2 "
  source env/bin/activate
  export PYTHONPATH="/home/nikita/MIT_6_4212/manipulation:${PYTHONPATH}"
  # Kill other jupyter instances. This might fail if those are not yours
  # pkill -f 'jupyter-notebook'
  # Spin it!
  ~/env/bin/jupyter-notebook --no-browser --port=8080
"


