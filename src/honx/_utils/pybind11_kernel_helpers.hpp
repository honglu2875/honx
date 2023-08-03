//
// Created by honglu on 8/3/23.
//

#ifndef _PYBIND11_KERNEL_HELPERS_H_
#define _PYBIND11_KERNEL_HELPERS_H_

#include <pybind11/pybind11.h>

namespace helper {

    template <class To, class From>
    typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
            To>::type
    bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible<To>::value,
    "This implementation additionally requires destination type to "
    "be trivially constructible");

    To dst;
    memcpy(&dst, &src, sizeof(To));
    return dst;
}
template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
    return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

} // namespace helper

#endif