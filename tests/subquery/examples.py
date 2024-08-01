# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

query_examples = """
1. 

# Provide me a list of addresses that have made the highest number of transfers to a specific address

query MyQuery {
  account(id: "0x0000000000000000000000000000000000000000") {
    id
    receivedTransfers {
      groupedAggregates(groupBy: FROM_ID) {
        keys
        distinctCount {
          id
        }
      }
    }
  }
}

2. 

# Please provide a list of addresses who transfered the highest amounts within the certain timeframe.

query MyQuery {
  account(id: "0x0000000000000000000000000000000000000000") {
    id
    receivedTransfers(
      first: 5
      filter: {and: [{timestamp: {greaterThan: "0"}}, {timestamp: {lessThan: "1000"}}]}
    ) {
      groupedAggregates(groupBy: FROM_ID) {
        keys
        sum {
          value
        }
      }
    }
  }
}
    
3. 

# Please provide a first transfer ever indexed

query MyQuery {
  transfers(first: 1, orderBy: TIMESTAMP_ASC) {
    nodes {
      id
      value
    }
  }
}
"""