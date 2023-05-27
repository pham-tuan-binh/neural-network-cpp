CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -O2 -std=c++20 -Ilib/Matrix -Ilib/NeuralNetwork -Ilib/econio -Ilib/Controller
LDLIBS = -lm

SRC_DIR = .
BUILD_DIR = build

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

LIB_SRCS := $(wildcard lib/Matrix/*.cpp) $(wildcard lib/NeuralNetwork/*.cpp) $(wildcard lib/econio/*.cpp) $(wildcard lib/Controller/*.cpp)
LIB_OBJS := $(LIB_SRCS:%.cpp=$(BUILD_DIR)/%.o)

.PHONY: all clean

all: $(BUILD_DIR)/my_program

$(BUILD_DIR)/my_program: $(OBJS) $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIB_OBJS) -o $@ $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: lib/Matrix/%.cpp | $(BUILD_DIR)/lib/Matrix
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: lib/NeuralNetwork/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: lib/econio/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: lib/Controller/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/lib/Matrix:
	mkdir -p $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/lib

clean:
	find $(BUILD_DIR) -type f -delete
