import os
import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def create_today_folder(base_path):
    # 获取今天的日期
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # 构建今天日期对应的文件夹路径
    today_folder_path = os.path.join(base_path, today_date)
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(today_folder_path):
        os.makedirs(today_folder_path)
    return today_folder_path
def make_prefix(seed, ansatz, no, e_name):
    base_path = "tmp/test/"
    today_folder_path = create_today_folder(base_path)
    
    name_fold = ansatz
    if seed is not None:
        name_fold += "_seed" + str(seed)
    name_fold += "_"+ str(e_name)
    if no is not None:
        name_fold += "_" + no
    
    pre_flod = str(today_folder_path)
    save_flod_0 = os.path.join(pre_flod,name_fold)
    if not os.path.exists(save_flod_0):
        os.makedirs(save_flod_0)
    save_flod = os.path.join(save_flod_0,name_fold)
    return(save_flod)

import numpy as np
def plot_figure(e_ref, e_lst, dataset, prefix):
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        e = np.array(dataset['energy'])
        idx = 0
        idx_e = np.arange(len(e))
        ax.plot(idx_e[idx:], e[idx:])
        ax.set_title(os.path.split(prefix)[1])  # remove path
        # ax.set_xlabel("Iteration Time")
        ax.set_ylabel(r"$\mathrm{Energy}_n$")
        if e_ref is not None:
            ax.axhline(e_ref, color="coral", ls="--")
            if e_lst is not None:
                for i in range(len(e_lst)):
                    ax.axhline(e_lst[i], color=plt.get_cmap("Accent")(i), ls="--")
            # plot partial enlarged view
            axins = inset_axes(
                ax,
                width="50%",
                height="45%",
                loc=1,
                bbox_to_anchor=(0.12, 0.12, 0.8, 0.8),
                bbox_transform=ax.transAxes,
            )
            axins.plot(e[idx:])
            axins.axhline(e_ref, color="coral", ls="--")
            if e_lst is not None:
                for i in range(len(e_lst)):
                    axins.axhline(e_lst[i], color=plt.get_cmap("Accent")(i), ls="--")
            zone_left = len(e) - len(e) // 10
            zone_right = len(e) - 1
            x_ratio = 0
            y_ratio = 1
            xlim0 = idx_e[zone_left] - (idx_e[zone_right] - idx_e[zone_left]) * x_ratio
            xlim1 = idx_e[zone_right] + (idx_e[zone_right] - idx_e[zone_left]) * x_ratio
            y = e[zone_left:zone_right]
            ylim0 = e_ref - (np.min(y) - e_ref) * y_ratio
            ylim1 = np.max(y) + (np.min(y) - e_ref) * y_ratio
            axins.set_xlim(xlim0, xlim1)
            axins.set_ylim(ylim0, ylim1)
            error = np.log(np.abs((e-e_ref)/e_ref))
            ax2 = ax.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel(r'$\mathrm{Error}_n = \log\left|\dfrac{E_n-E_{\rm ref}}{E_n}\right|$', color=color2)  # we already handled the x-label with ax1
            ax2.plot(idx_e[idx:], error[idx:], color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            # ax2.set_yscale("log")
            axins2= axins.twinx()
            # axins2.set_ylabel(r'$\mathrm{Error}_n = \log\left|\dfrac{E_n-E_{\rm ref}}{E_n}\right|$', color=color2)  # we already handled the x-label with ax1
            axins2.plot(idx_e[idx:], error[idx:], color=color2)
            axins2.tick_params(axis='y', labelcolor=color2)
        # plot the L2-norm and max-abs of the gradients
        param_L2 = dataset['grad_L2']
        param_max = dataset['grad_max']

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(np.arange(len(param_L2))[idx:], param_L2[idx:], label="||g||")
        ax.plot(np.arange(len(param_max))[idx:], param_max[idx:], label="max|g|")
        ax.set_xlabel(r"$\mathrm{Iteration Time} n$")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\mathrm{Gradients}_n$")
        plt.legend(loc="best")

        # plt.subplots_adjust(wspace=0, hspace=0.5)
        # plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        fig.tight_layout()
        plt.savefig(prefix + ".pdf")
        plt.close()
def plot_figure_error(e_ref, e_lst, dataset1, data1, dataset2, data2, prefix):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    def plote(dataset,data):
        e = np.array(dataset['energy'])
        idx = 0
        idx_e = np.arange(len(e))
        ax.plot(idx_e[idx:], e[idx:],label=data)
    plote(dataset1,data1)
    plote(dataset2,data2)
    ax.set_title(os.path.split(prefix)[1])  # remove path
    # ax.set_xlabel("Iteration Time")
    ax.set_ylabel(r"$\mathrm{Energy}_n$")
    if e_ref is not None:
        ax.axhline(e_ref, color="coral", ls="--")
        if e_lst is not None:
            for i in range(len(e_lst)):
                ax.axhline(e_lst[i], color=plt.get_cmap("Accent")(i), ls="--")
        # plot partial enlarged view
        # axins = inset_axes(
        #     ax,
        #     width="50%",
        #     height="45%",
        #     loc=1,
        #     bbox_to_anchor=(0.12, 0.12, 0.8, 0.8),
        #     bbox_transform=ax.transAxes,
        # )
        # axins.plot(e[idx:])
        # axins.axhline(e_ref, color="coral", ls="--")
        # if e_lst is not None:
        #     for i in range(len(e_lst)):
        #         axins.axhline(e_lst[i], color=plt.get_cmap("Accent")(i), ls="--")
        # zone_left = len(e) - len(e) // 10
        # zone_right = len(e) - 1
        # x_ratio = 0
        # y_ratio = 1
        # xlim0 = idx_e[zone_left] - (idx_e[zone_right] - idx_e[zone_left]) * x_ratio
        # xlim1 = idx_e[zone_right] + (idx_e[zone_right] - idx_e[zone_left]) * x_ratio
        # y = e[zone_left:zone_right]
        # ylim0 = e_ref - (np.min(y) - e_ref) * y_ratio
        # ylim1 = np.max(y) + (np.min(y) - e_ref) * y_ratio
        # axins.set_xlim(xlim0, xlim1)
        # axins.set_ylim(ylim0, ylim1)
        # plt.legend(loc="best")


       
        # color2 = 'tab:red'
        
        # ax2.tick_params(axis='y', labelcolor=color2)
        # ax2.set_yscale("log")
        # axins2= axins.twinx()
        # axins2.set_ylabel(r'$\mathrm{Error}_n = \log\left|\dfrac{E_n-E_{\rm ref}}{E_n}\right|$', color=color2)  # we already handled the x-label with ax1
        # axins2.plot(idx_e[idx:], error[idx:], color=color2)
        # axins2.tick_params(axis='y', labelcolor=color2)
        # 绘制误差对数
        ax2 = fig.add_subplot(2, 1, 2)
        def plotr(dataset,data):
            e = np.array(dataset['energy'])
            error = np.log(np.abs((e-e_ref)/e_ref))
            idx = 0
            idx_e = np.arange(len(e))
            ax2.plot(idx_e[idx:], error[idx:],label=data)
        plotr(dataset1,data1)
        plotr(dataset2,data2)
        ax2.set_ylabel(r'$\mathrm{Error}_n = \log\left|\dfrac{E_n-E_{\rm ref}}{E_n}\right|$')  # we already handled the x-label with ax1
        # plt.legend(loc="best")
    plt.legend(loc="best")
    fig.tight_layout()
    plt.savefig("error" + ".pdf")
    plt.close()
