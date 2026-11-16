#include <cstdint>
#include "Request.h"

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

void Request::Request::deallocateMemory(GPU &gpu) 
{
    
}

std::size_t Request::getNeededMemory(const GPU &gpu) const 
{

}

std::size_t Request::getMaxNeededMemory() const {}

std::vector<std::string> Request::getCacheableTokens() const {}

void Request::allocateOneBlock(GPU &gpu, std::size_t slot) const {}

Request *Request::clone(const Model &model) const {}

void Request::act(GPU &gpu) {}

bool Request::finished() const {}

void Request::reserveMemory(GPU &gpu) 
{
    
}

std::string Request::getOutput() const {}

std::size_t Request::getNeedToUnreserve() const {}

GenerativeRequest::GenerativeRequest(int id, const Model &model, int admissionCycle, std::string prompt, std::string scriptedOutput, std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens)
{
}

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt)
{
}