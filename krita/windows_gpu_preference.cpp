/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifdef _WIN32
// Optimus reads this executable export before OpenGL initialization. Request
// the discrete GPU for the compute brush path on supported hybrid laptops.
// Windows/user graphics preferences can override this request.
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 1;
}
#endif
