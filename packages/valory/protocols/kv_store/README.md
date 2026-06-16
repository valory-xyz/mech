# Key-Value Storage Protocol

## Description

This is a protocol for key-value storage.

## Specification

```yaml
---
name: kv_store
author: valory
version: 0.1.0
description: A protocol for simple key-value storage.
license: Apache-2.0
aea_version: '>=1.0.0, <2.0.0'
protocol_specification_id: valory/kv_store:0.1.0
speech_acts:
  read_request:
    keys: pt:list[pt:str]
  read_response:
    data: pt:dict[pt:str, pt:str]
  create_or_update_request:
    data: pt:dict[pt:str, pt:str]
  delete_request:
    keys: pt:list[pt:str]
  list_request:
    key_prefix: pt:str
  list_response:
    data: pt:dict[pt:str, pt:str]
  success:
    message: pt:str
  error:
    message: pt:str
...
---
initiation: [read_request, create_or_update_request, delete_request, list_request]
reply:
  read_request: [read_response, error]
  read_response: []
  create_or_update_request: [success, error]
  delete_request: [success, error]
  list_request: [list_response, error]
  list_response: []
  success: []
  error: []
termination: [read_response, list_response, success, error]
roles: {skill, connection}
end_states: [successful]
keep_terminal_state_dialogues: false
```

## Links

