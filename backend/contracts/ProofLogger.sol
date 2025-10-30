// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ProofLogger {
    event ProofLogged(address indexed sender, string proof);

    struct Log {
        address sender;
        string proof;
        uint256 timestamp;
    }

    Log[] public logs;

    function logProof(string memory proof) public {
        logs.push(Log(msg.sender, proof, block.timestamp));
        emit ProofLogged(msg.sender, proof);
    }

    function getLogsCount() public view returns (uint256) {
        return logs.length;
    }

    function getLog(uint256 index) public view returns (address, string memory, uint256) {
        require(index < logs.length, "Invalid index");
        Log memory entry = logs[index];
        return (entry.sender, entry.proof, entry.timestamp);
    }
}
