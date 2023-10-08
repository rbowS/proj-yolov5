#include "simple-logger.cuh"
#include <string>
#include <stdarg.h>

using namespace std;

namespace SimpleLogger{

    static LogLevel g_level = LogLevel::Info;

    const char* level_string(LogLevel level){
        switch (level){
            case LogLevel::Debug: return "debug";
            case LogLevel::Verbose: return "verbo";
            case LogLevel::Info: return "info";
            case LogLevel::Warning: return "warn";
            case LogLevel::Error: return "error";
            case LogLevel::Fatal: return "fatal";
            default: return "unknow";
        }
    }

    void set_log_level(LogLevel level){
        g_level = level;
    }

    LogLevel get_log_level(){
        return g_level;
    }

    //根据一个文件路径（path），返回一个文件名（string）。
    //例如，如果path为"/home/user/simple-logger.cpp"，则返回"simple-logger.cpp"；
    string file_name(const string& path, bool include_suffix){

        if (path.empty()) return "";

        int p = path.rfind('/');
        p += 1;

        //include suffix
        if (include_suffix)
            return path.substr(p);

        int u = path.rfind('.');
        if (u == -1)
            return path.substr(p);

        if (u <= p) u = path.size();
        return path.substr(p, u - p);
    }

    //获取当前的时间（time_t）并转换为一个格式化的字符串（string）。
    //例如，如果当前时间为2023年10月3日21点23分06秒，则返回"2023-10-03 21:23:06"
    string time_now(){
        char time_string[20];
        time_t timep;							
        time(&timep);							
        tm& t = *(tm*)localtime(&timep);

        sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
        return time_string;
    }

    //根据不同的日志级别（level），输出文件名（file），行号（line），以及其他参数（fmt, …）到标准输出（stdout）。
    void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...){
        if(level > g_level) return;

        //使用va_list类型和va_start函数，来获取可变参数的列表（vl）
        va_list vl;
        va_start(vl, fmt);
        
        char buffer[2048];
        auto now = time_now();
        string filename = file_name(file, true);
        int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());

        //使用一系列的snprintf函数，将当前时间（now），日志级别（level_string(level)），
        //文件名（filename），行号（line），以及其他参数（fmt, vl）格式化为字符串，并拼接到buffer中
        if (level == LogLevel::Fatal or level == LogLevel::Error) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Warning) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Info) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]", level_string(level));
        }
        else if (level == LogLevel::Verbose) {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[34m%s\033[0m]", level_string(level));
        }
        else {
            n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]", level_string(level));
        }

        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]:", filename.c_str(), line);
        vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
        fprintf(stdout, "%s\n", buffer);

        //将buffer中的内容输出到标准输出，并换行。如果level为LogLevel::Fatal或LogLevel::Error，
        //则还要使用fflush函数，强制刷新标准输出，并使用abort函数，终止程序的运行
        if(level == LogLevel::Fatal || level == LogLevel::Error){
            fflush(stdout);
            abort();
        }
    }
};
