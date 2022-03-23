#version 450 compatibility
#extension GL_ARB_shader_image_load_store : enable
#include "/lib/header.glsl"
#define WORLD WORLD_OVERWORLD
#define STAGE STAGE_FRAGMENT
#include "/program/gbuffers_basic.glsl"
