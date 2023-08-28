#ifndef __AS_TOOLS__
#define __AS_TOOLS__
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef VERBOSE_MODE
#define VERBOSE_MODE -1
#endif

// Macro for output of information, warning and error messages

#if VERBOSE_MODE >= 0
  #ifdef FNAME 
    #define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
    #define AS_WARNING(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf("<[AS_WARN]: %s,%04i> " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }
    #define AS_WARNING_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf("<[AS_WARN]: %s,%04i> " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }
    
    #define AS_ERROR(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        fprintf(stderr, "<[AS_ERRO]: %s,%04i> " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }
    #define AS_ERROR_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        fprintf(stderr, "<[AS_ERRO]: %s,%04i> " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }
    
    #define AS_INFOMSG(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf("<[AS_INFO]: %s,%04i> " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf("<[AS_INFO]: %s,%04i> " message "\n", __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }

    #define AS_INFOMSG1(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf("<[AS_INFO]: %s,%04i> " message, __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG1_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf("<[AS_INFO]: %s,%04i> " message, __FILENAME__, __LINE__, ##__VA_ARGS__); \
      } \
    }

    #define AS_INFOMSG2(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf(message, ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG2_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf(message, ##__VA_ARGS__); \
      } \
    }
  #else
    // #define __FILENAME__ ""
    #define AS_WARNING(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf("[AS_WARN]: " message "\n", ##__VA_ARGS__); \
      } \
    }
    #define AS_WARNING_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf("[AS_WARN]: " message "\n", ##__VA_ARGS__); \
      } \
    }
    
    #define AS_ERROR(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        fprintf(stderr, "[AS_ERRO]: " message "\n", ##__VA_ARGS__); \
      } \
    }
    #define AS_ERROR_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        fprintf(stderr, "[AS_ERRO]: " message "\n", ##__VA_ARGS__); \
      } \
    }
    
    #define AS_INFOMSG(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf("[AS_INFO]: " message "\n", ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf("[AS_INFO]: " message "\n", ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG1(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf("[AS_INFO]: " message, ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG1_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf("[AS_INFO]: " message, ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG2(level, message, ...) { \
      if(VERBOSE_MODE >= level) {\
        printf(message, ##__VA_ARGS__); \
      } \
    }
    #define AS_INFOMSG2_IF(level, condition, message, ...) { \
      if(VERBOSE_MODE >= level && condition) { \
        printf( message, ##__VA_ARGS__); \
      } \
    }

  #endif
#else
  #define AS_WARNING(level, message, ...) {}
  #define AS_WARNING_IF(level, message, ...) {}
  #define AS_ERROR(level, message, ...) {}
  #define AS_ERROR_IF(level, message, ...) {}
  #define AS_INFOMSG(level, message, ...) {}
  #define AS_INFOMSG_IF(level, message, ...) {}
  #define AS_INFOMSG1(level, message, ...) {}
  #define AS_INFOMSG_IF1(level, message, ...) {}
  #define AS_INFOMSG2(level, message, ...) {}
  #define AS_INFOMSG_IF2(level, message, ...) {}
#endif // END IF VERBOSE_MODE


#endif // __AS_TOOLS__
