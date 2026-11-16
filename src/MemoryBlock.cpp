#include <cstdint>
#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size)
{
    this->size = size;
    this->data = new int[size];
    this->refCount = 0;
}

int MemoryBlock::readFromBlock(std::size_t offset) const
{
    return *(data + offset);
}

void MemoryBlock::writeToBlock(std::size_t offset, int value)
{
    *(data + offset) = value;
}

void MemoryBlock::acquire()
{
    refCount++;
}

IdleAction MemoryBlock::release()
{
    refCount--;
    if (refCount == 0)
        return onLastRelease();
}

IdleAction KVBlock::onLastRelease() const
{
    return IdleAction::KeepCached;
}

MemoryBlock *KVBlock::clone() const
{
    MemoryBlock *mb = new KVBlock(size);
    for (size_t i = 0; i < size; i++)
        mb->writeToBlock(i, readFromBlock(i));
    return mb;
}

IdleAction EmbeddingWorkspace::onLastRelease() const
{
    return IdleAction::Destroy;
}

MemoryBlock *EmbeddingWorkspace::clone() const
{
        MemoryBlock *mb = new EmbeddingWorkspace(size);
    for (size_t i = 0; i < size; i++)
        mb->writeToBlock(i, readFromBlock(i));
    return mb;
}
