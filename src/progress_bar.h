/*
Credit to Prakhar Srivastav for the initial
version of the progress bar, available at:
https://github.com/prakhar1989/progress-cpp
The code below contains changes,
for ease of use of the JBU method.
*/

#ifndef __PROGRESS_BAR__
#define __PROGRESS_BAR__


#include <chrono>
#include <iostream>
#include <RcppArmadillo.h>


class ProgressBar {

private:
    unsigned int ticks = 0;

    const unsigned int total_ticks;
    const unsigned int bar_width;
    const char complete_char = '=';
    const char incomplete_char = ' ';
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();


public:
    ProgressBar(unsigned int total, unsigned int width, char complete, char incomplete) :
            total_ticks{total}, bar_width{width}, complete_char{complete}, incomplete_char{incomplete} {}

    ProgressBar(unsigned int total, unsigned int width) : total_ticks{total}, bar_width{width} {}

    unsigned int operator++() { return ++ticks; }

    void display() const {
        float progress = (float) ticks / total_ticks;
        int pos = (int) (bar_width * progress);

        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        std::cout << "[";

        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << complete_char;
            else if (i == pos) std::cout << ">";
            else std::cout << incomplete_char;
        }
        int cent_secs = round(float(time_elapsed) / 10.0);
        int secs = floor(float(time_elapsed) / 1000.0);
        int mins = floor(float(time_elapsed) / 60000.0);
        int hours = floor(float(time_elapsed) / 3600000.0);
        std::cout << "] " << int(progress * 100.0) << "%  -  "
                  << hours << "h "
                  << int(mins - floor(mins / 60.0) * 60.0) << "m ";
        int secs_out = int(secs - floor(secs / 60.0) * 60.0);
        if (secs_out < 10) {
          std::cout << " " << secs_out << ".";
        } else {
          std::cout << secs_out << ".";
        }
        std::cout << int(cent_secs - floor(cent_secs / 100.0) * 100.0) << "s\r";
        std::cout.flush();
    }

    void done() const {
        display();
        std::cout << std::endl;
    }
};

#endif
