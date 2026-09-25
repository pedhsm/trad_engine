cmake_minimum_required(VERSION 3.16)
project(trad_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Interactive Brokers TWS API client -------------------------------------
# NOT shipped with this repo (IB's API is not redistributable). You supply it in
# ../lib/client — see lib/client/README.md for how to populate it.
set(IB_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/client")
# TwsApiL0.cpp is a unity build: it #includes EPosixClientSocket.cpp and
# EClientSocketBase.cpp itself, so it is the ONLY client source to compile (globbing
# *.cpp would compile those twice and fail at link time with duplicate symbols).
set(IB_CLIENT_SOURCES "${IB_CLIENT_DIR}/TwsApiL0.cpp")
if(NOT EXISTS "${IB_CLIENT_SOURCES}")
    message(FATAL_ERROR
        "${IB_CLIENT_SOURCES} not found. Populate lib/client with the IB TWS API / "
        "TwsApiC++ before building — see lib/client/README.md.")
endif()

add_executable(trad_engine
    src/main.cpp
    src/AllocatorArena.cpp
    ${IB_CLIENT_SOURCES}
)

# std::string instead of MFC's CString inside the IB client (the engine header sets
# it too, but TwsApiL0.cpp is compiled on its own).
target_compile_definitions(trad_engine PRIVATE IB_USE_STD_STRING $<$<PLATFORM_ID:Windows>:WIN32>)

target_include_directories(trad_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${IB_CLIENT_DIR}
)

# --- Threads ----------------------------------------------------------------
find_package(Threads REQUIRED)
target_link_libraries(trad_engine PRIVATE Threads::Threads)

# --- ZeroMQ (IPC transport) -------------------------------------------------
# Prefer the cppzmq CMake package; fall back to linking libzmq directly (make sure
# zmq.hpp is on the include path in that case).
find_package(cppzmq QUIET)
if(cppzmq_FOUND)
    target_link_libraries(trad_engine PRIVATE cppzmq)
else()
    target_link_libraries(trad_engine PRIVATE zmq)
endif()

# --- Windows sockets --------------------------------------------------------
if(WIN32)
    target_link_libraries(trad_engine PRIVATE ws2_32)
endif()
