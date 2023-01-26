#include <loguru/loguru.hpp>

#include "nerf-lab/gui/app.h"

int main(int argc, char** argv) {
    loguru::init(argc, argv);
    loguru::set_fatal_handler([](const loguru::Message& msg) {
        throw std::runtime_error(std::string(msg.prefix) + msg.message);
    });

    nerf::App().run();
    return 0;
}