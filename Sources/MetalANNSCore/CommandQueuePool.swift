import Metal

/// A fixed-size pool of MTLCommandQueues for pipelining GPU work across concurrent searches.
package actor CommandQueuePool: Sendable {
    package let queues: [MTLCommandQueue]
    private var nextIndex: Int = 0

    package init(device: MTLDevice, count: Int = 4) throws {
        guard count > 0 else {
            throw ANNSError.constructionFailed("Command queue pool count must be > 0")
        }

        var createdQueues: [MTLCommandQueue] = []
        createdQueues.reserveCapacity(count)

        for _ in 0..<count {
            guard let queue = device.makeCommandQueue() else {
                throw ANNSError.deviceNotSupported
            }
            createdQueues.append(queue)
        }

        self.queues = createdQueues
    }

    /// Round-robin queue selection. Thread-safe via actor isolation.
    package func next() -> MTLCommandQueue {
        let queue = queues[nextIndex % queues.count]
        nextIndex &+= 1
        return queue
    }
}
