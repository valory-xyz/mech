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