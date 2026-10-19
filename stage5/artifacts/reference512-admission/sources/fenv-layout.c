#include <fenv.h>
#include <stdio.h>
int main(void) {
    printf("{\"sizeof_fenv\": %zu, \"alignment\": %zu, \"nearest\": %d}\n",
           sizeof(fenv_t), _Alignof(fenv_t), FE_TONEAREST);
}
