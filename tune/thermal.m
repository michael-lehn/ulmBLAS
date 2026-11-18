/*
 *  thermal: prints the thermal state of macOS (NSProcessInfo.thermalState:
 *  nominal, fair, serious or critical) and, with an interval in seconds,
 *  keeps printing it with a time stamp (thermal SECONDS).
 */
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int
main(int argc, char **argv)
{
    static const char *name[] = { "nominal", "fair", "serious", "critical" };
    int interval = argc>1 ? atoi(argv[1]) : 0;
    do {
        long s = [[NSProcessInfo processInfo] thermalState];
        char t[32];
        time_t now = time(0);
        strftime(t, sizeof t, "%H:%M:%S", localtime(&now));
        printf("%s thermal %s\n", t, (s>=0 && s<4) ? name[s] : "unknown");
        fflush(stdout);
        if (interval>0) sleep(interval);
    } while (interval>0);
    return 0;
}
