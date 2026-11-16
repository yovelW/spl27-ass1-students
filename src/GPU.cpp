#include <cstdint>
#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include "cmath"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
{
    this->blocksNum = blocksNum;
    this->blockSize = blockSize;
    memory = std::vector<MemoryBlock*>(blocksNum);
}

std::size_t GPU::availableSlots() const
{
    int ActiveBlocks = 0;
    for (size_t i = 0; i < blocksNum; i++)
    {
        if(memory[i]); //todo fix or some bs idk
    }
    
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const
{
    return ceil(memory/blockSize);
}

void GPU::reserve(std::size_t n, const Request &request)
{

}

void GPU::unreserve(std::size_t n, const Request &request){}

std::size_t GPU::allocateBlock(const Request &request)
{
}

void GPU::addMemoryBlock(const GenerativeRequest &request, std::size_t slot){}

void GPU::addMemoryBlock(const EmbeddingRequest &request, std::size_t slot){}

void GPU::acquire(std::size_t slot){}

void GPU::release(std::size_t slot){}

void GPU::copyMemoryFromOtherGPU(const GPU &other, const std::vector<std::size_t> &blocks){}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string> &prefixTokens, const std::string &modelId){}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const{}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value){}

PrefixMatch GPU::findLongestCachedPrefix(const std::string &modelId, std::size_t embeddingSize, const std::vector<std::string> &promptTokens) const{}