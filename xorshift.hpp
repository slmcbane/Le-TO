#ifndef XORSHIFT_HPP
#define XORSHIFT_HPP

#include <cstdint>
#include <limits>
#include <random>

class Xorshift64StarEngine
{
public:
    typedef uint64_t result_type;

    constexpr uint64_t min() { return 0; }
    constexpr uint64_t max() { return std::numeric_limits<uint64_t>::max(); }

    constexpr static uint64_t mult_constant = 0x2545F4914F6CDD1DULL;

    uint64_t operator()()
    {
        m_state ^= m_state >> 12;
        m_state ^= m_state << 25;
        m_state ^= m_state >> 27;
        return m_state * mult_constant;
    }

    static_assert(sizeof(std::random_device::result_type) == sizeof(uint64_t) ||
                  sizeof(std::random_device::result_type) * 2 == sizeof(uint64_t));

    Xorshift64StarEngine()
    {
        if constexpr (sizeof(std::random_device::result_type) == sizeof(uint64_t))
        {
            m_state = std::random_device()();
        }
        else
        {
            m_state = 0;
            std::random_device dev;
            m_state |= static_cast<uint64_t>(dev());
            m_state |= static_cast<uint64_t>(dev()) << 32;
        }
    }

    Xorshift64StarEngine(uint64_t seed) noexcept : m_state(seed) {}

private:
    uint64_t m_state;
};

#endif // XORSHIFT_HPP
