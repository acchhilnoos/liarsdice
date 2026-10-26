UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
  OPT_FLAGS := -mavx2 -mfma
else
  OPT_FLAGS := -ffast-math
endif

CC := gcc
CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Iinclude
LIBS := -lm

SRCDIR  := src
OBJDIR  := build
TESTDIR := tests

SRCS := $(wildcard $(SRCDIR)/*.c)
OBJS := $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SRCS))
TESTOBJS := $(OBJDIR)/game.o $(OBJDIR)/network.o $(OBJDIR)/tensor.o $(OBJDIR)/test.o

TARGET := main

all: CFLAGS += -O3 $(OPT_FLAGS)
all: $(TARGET)

debug: CFLAGS += -O0 -g $(OPT_FLAGS)
debug: $(TARGET)

test: CFLAGS += -O3 $(OPT_FLAGS)
test: $(TESTOBJS)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ $(LIBS)

$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $^ -o $@ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/test.o: $(TESTDIR)/test.c | $(OBJDIR)
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	@mkdir $@

clean:
	@rm -rf $(OBJDIR) $(TARGET) test
	@rm -rf *.dSYM

.PHONY: all clean debug test
