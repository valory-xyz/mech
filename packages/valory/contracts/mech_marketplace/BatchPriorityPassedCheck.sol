// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

struct MechDelivery {
    // Priority mech address
    address priorityMech;
    // Delivery mech address
    address deliveryMech;
    // Requester address
    address requester;
    // Response timeout window
    uint32 responseTimeout;
}

interface IMechMarketplace {
    function mapRequestIdDeliveries(uint256) external view returns (MechDelivery);
}

contract BatchPriorityPassedCheck {
    constructor(IMechMarketplace _marketplace, uint256[] memory _requestIds) {
        // cache requestIds length
        uint256 requestIdsLength = _requestIds.length;

        // create temporary array with requestIds length to populate only with requestIds that have passed the priority timeout
        address[] memory tempRequestIds = new address[](requestIdsLength);

        // declare counter to know how many of the request are eligible
        uint256 eligibleRequestIdsCount;

        for (uint256 _i; _i < requestIdsLength;) {
            MechDelivery memory delivery = _marketplace.mapRequestIdDeliveries(_requestIds[_i]);
            if (block.timestamp >= delivery.responseTimeout) {
                tempRequestIds[eligibleRequestIdsCount] = _requestIds[_i];
                ++eligibleRequestIdsCount;
            }
            unchecked {++_i;}
        }

        // create a new array with the actual length of the eligible to not corrupt memory with a wrong length
        address[] memory eligibleRequestIds = new address[](eligibleRequestIdsCount);

        // populate the array with the eligible requestIds
        for (uint256 _i; _i < eligibleRequestIdsCount;) {
            eligibleRequestIds[_i] = tempRequestIds[_i];
            unchecked {++_i;}
        }

        // encode eligible referrers to ensure a proper layout in memory
        bytes memory data = abi.encode(eligibleRequestIds);

        assembly {
            // pointer to the beginning of the data containing the eligible referrers in memory
            let _dataStartPointer := add(data, 32)
            // return everything from the start of the data to the end of memory
            return (_dataStartPointer, sub(msize(), _dataStartPointer))
        }
    }

}