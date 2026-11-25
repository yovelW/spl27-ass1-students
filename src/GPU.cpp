#include <cstdint>
#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include "cmath"
#include <stdexcept>

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
{
    this->blocksNum = blocksNum;
    this->blockSize = blockSize;
    memory = std::vector<MemoryBlock *>(blocksNum);
}

std::size_t GPU::availableSlots() const
{
    int activeBlocks = 0;
    for (size_t i = 0; i < blocksNum; i++)
    {
        if (!memory[i]->isIdle())
            activeBlocks++;
    }

    return blocksNum - activeBlocks - resNum;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const
{
    return ceil(memory / blockSize);
}

void GPU::reserve(std::size_t n, const Request &request)
{
    if (n > availableSlots())
        throw std::invalid_argument("not enough blocks to reserve");

    requests[request.getId()] += n;
    resNum += n;
}

void GPU::unreserve(std::size_t n, const Request &request)
{
    if (requests[request.getId()] < n)
        throw std::invalid_argument("not enough blocks to unreserve");

    requests[request.getId()] -= n;
    resNum -= n;
}

std::size_t GPU::allocateBlock(const Request &request)
{
    size_t slot = getNextBlock();

    request.allocateOneBlock(*this, slot);
    acquire(slot);

    resNum--;
    return slot;
}

void GPU::addMemoryBlock(const GenerativeRequest &request, std::size_t slot)
{
    KVBlock *block = new KVBlock(blockSize);
    memory[slot] = block;

    requests[request.getId()]--;
}

void GPU::addMemoryBlock(const EmbeddingRequest &request, std::size_t slot)
{
    EmbeddingWorkspace *block = new EmbeddingWorkspace(blockSize);
    memory[slot] = block;

    requests[request.getId()]--;
}

void GPU::acquire(std::size_t slot)
{
    memory[slot]->acquire();
}

void GPU::release(std::size_t slot)
{
    memory[slot]->release();
}

void GPU::copyMemoryFromOtherGPU(const GPU &other, const std::vector<std::size_t> &blocks)
{
    for (size_t i = 0; i < blocks.size(); i++)
        memory[getNextBlock()] = other.memory[blocks[i]]->clone();
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string> &prefixTokens, const std::string &modelId)
{
    models[block] = modelId;
    prefixes[block] = prefixTokens;
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const
{
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value)
{
    memory[blockId]->writeToBlock(offset, value);
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string &modelId, std::size_t embeddingSize, const std::vector<std::string> &promptTokens) const
{
    PrefixMatch result;
    result.matchedTokens = 0;

    if (embeddingSize == 0 || promptTokens.empty())
        return result;

    std::size_t tokensPerBlock = blockSize / embeddingSize;
    if (tokensPerBlock == 0)
        return result;

    std::vector<std::string> currentPrefix;
    currentPrefix.reserve(promptTokens.size());

    for (std::size_t len = tokensPerBlock; len <= promptTokens.size(); len += tokensPerBlock)
    {
        for (std::size_t k = len - tokensPerBlock; k < len; k++)
        {
            currentPrefix.push_back(promptTokens[k]);
        }

        bool found = false;
        for (std::size_t slot = 0; slot < blocksNum; slot++)
        {
            if (memory[slot] != nullptr && models[slot] == modelId && prefixes[slot] == currentPrefix)
            {
                result.blockIds.push_back(slot);
                result.matchedTokens = len;
                found = true;
                break;
            }
        }

        if (!found)
            break;
    }

    return result;
}

std::size_t GPU::getNextBlock()
{
    size_t slot = -1;

    for (size_t i = 0; i < blocksNum; i++)
        if (memory[i] == nullptr)
            return slot;

    for (size_t i = 0; i < blocksNum; i++)
        if (memory[i]->isIdle())
        {
            delete memory[i];
            memory[i] = nullptr; //??
            return slot;
        }
}

std::size_t GPU::getBlockSize() const
{
    return blockSize;
}