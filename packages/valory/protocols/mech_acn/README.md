# ACN data request protocol for mechs

## Description

This protocol is for performing data requests on the mech agents.

## Specification

```yaml
---
name: mech_acn
author: valory
version: 0.1.0
description: A protocol for adding ACN callbacks on AI mechs.
license: Apache-2.0
aea_version: '>=1.0.0, <2.0.0'
protocol_specification_id: valory/mech_acn:0.1.0
speech_acts:
  request:
    request_id: pt:str
  response:
    data: pt:str
    status: ct:Status
...
---
ct:Status: |
  enum Status {
      REQUEST_NOT_FOUND = 0;
      DATA_NOT_READY = 1;
      READY = 2;
      REQUEST_EXPIRED = 3;
    }
  Status status = 1;
...
---
initiation: [request]
reply:
  request: [response]
  response: []
termination: [response]
roles: {agent,skill}
end_states: [successful, failed]
keep_terminal_state_dialogues: false
...
```

