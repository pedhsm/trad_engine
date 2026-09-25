cmake_minimum_required(VERSION 3.16)
project(trad_engine_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Interactive Brokers TWS API client -------------------------------------
# NOT shipped with this repo (IB's API is not redistributable). You supply it in
# ../lib/client — see lib/client/README.md for how to populate it.
set(IB_CLIENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/client")
file(GLOB IB_CLIENT_SOURCES "${IB_CLIENT_DIR}/*.cpp")
if(NOT IB_CLIENT_SOURCES)
    message(FATAL_ERROR
        "No .cpp sources found in ${IB_CLIENT_DIR}. Populate it with the IB TWS API / "
        "TwsApiC++ before building — see lib/client/README.md.")
endif()

add_executable(trad_engine
    src/main.cpp
    src/AllocatorArena.cpp
    ${IB_CLIENT_SOURCES}
)

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
