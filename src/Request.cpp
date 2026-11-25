#include <cstdint>
#include "Request.h"
#include "GPU.h"
#include "Model.h"

Request::Request(int id, const Model &model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), reservedBlocks(0)
{
}

int Request::getId() const
{
    return id;
}

const Model &Request::getModel() const
{
    return model;
}

int Request::getAdmissionCycle() const
{
    return admissionCycle;
}

void Request::deallocateMemory(GPU &gpu)
{
    std::size_t unused = getNeedToUnreserve();
    if (unused > 0)
    {
        gpu.unreserve(unused, *this);
        reservedBlocks = 0;
    }
    for (std::size_t slot : memoryBlocks)
    {
        gpu.release(slot);
    }
    memoryBlocks.clear();
}

Request *Request::clone(const Model &model) const {}

bool Request::finished() const {}

std::size_t Request::getNeedToUnreserve() const
{
    return reservedBlocks;
}

GenerativeRequest::GenerativeRequest(int id, const Model &model, int admissionCycle, std::string prompt, std::string scriptedOutput, std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens)
{
}

std::size_t GenerativeRequest::getNeededMemory(const GPU &gpu) const
{

    PrefixMatch prefix = gpu.findLongestCachedPrefix(model.getId(), model.getEmbedSize(), tokens);
    return model.getEmbedSize() * (tokens.size() - prefix.matchedTokens + maxOutputTokens);
}

std::size_t GenerativeRequest::getMaxNeededMemory() const
{
    return model.getEmbedSize() * (maxOutputTokens + tokens.size());
}

void GenerativeRequest::act(GPU &gpu)
{
    if (finished())
        return;

    std::size_t embedSize = model.getEmbedSize();
    std::size_t tpb = gpu.getBlockSize() / embedSize;
}

void GenerativeRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const
{
    gpu.addMemoryBlock(*this, slot);
}

std::vector<std::string> GenerativeRequest::getCacheableTokens() const
{
    return tokens;
}

std::string GenerativeRequest::getOutput() const
{
}

EmbeddingRequest::EmbeddingRequest(int id, const Model &model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt)
{
}

std::size_t EmbeddingRequest::getNeededMemory(const GPU &gpu) const
{
    return 2 * model.getEmbedSize();
}

std::size_t EmbeddingRequest::getMaxNeededMemory() const
{
    return 2 * model.getEmbedSize();
}

void EmbeddingRequest::act(GPU &gpu) {}

void EmbeddingRequest::allocateOneBlock(GPU &gpu, std::size_t slot) const
{
    gpu.addMemoryBlock(*this, slot);
}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const
{
    return {};
}

std::string EmbeddingRequest::getOutput() const
{
}