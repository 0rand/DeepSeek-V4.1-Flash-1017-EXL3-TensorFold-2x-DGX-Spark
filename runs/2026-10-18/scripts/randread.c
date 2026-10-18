// randread FILE THREADS SECONDS DIRECT(0/1) [BYTES]: random aligned preads from FILE, prints reads/s and mean latency.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int fd;
static off_t size;
static double secs;
static size_t bytes = 4096;
static volatile long total;
static volatile double lat_sum;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *run(void *arg) {
    uint64_t s = (uint64_t)(uintptr_t)arg * 2654435761u + 12345;
    void *buf;
    if (posix_memalign(&buf, 4096, bytes < 4096 ? 4096 : bytes)) return NULL;
    long n = 0;
    double lat = 0, end = now() + secs;
    while (1) {
        double t0 = now();
        if (t0 >= end) break;
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        off_t off = (off_t)(s % (uint64_t)(size / 4096 - 1)) * 4096;
        if (pread(fd, buf, bytes, off) < 0) { perror("pread"); break; }
        lat += now() - t0;
        n++;
    }
    pthread_mutex_lock(&mu);
    total += n; lat_sum += lat;
    pthread_mutex_unlock(&mu);
    free(buf);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: randread FILE THREADS SECONDS DIRECT [BYTES]\n"); return 2; }
    int th = atoi(argv[2]);
    secs = atof(argv[3]);
    int direct = atoi(argv[4]);
    if (argc > 5) bytes = (size_t)atol(argv[5]);
    fd = open(argv[1], O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) { perror("open"); return 1; }
    if (!direct) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    struct stat st;
    fstat(fd, &st);
    size = st.st_size;
    pthread_t t[1024];
    for (int i = 0; i < th; i++) pthread_create(&t[i], NULL, run, (void *)(uintptr_t)(i + 1));
    for (int i = 0; i < th; i++) pthread_join(t[i], NULL);
    printf("threads %d direct %d bytes %zu: %.0f reads/s, mean latency %.0f us\n", th, direct, bytes,
           total / secs, 1e6 * lat_sum / (total ? total : 1));
    return 0;
}
