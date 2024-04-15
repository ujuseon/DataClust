import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# Загрузка CSV-файла
def load_csv(file):
    df = pd.read_csv(file)
    return df

# ==== Функции для алгоритмов кластеризации ====
# Функция для K-means
def kmeans_clustering(df, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(df)
    return clusters

# Функция для K-means with outliers
def kmeans_with_ouliers(df, num_clusters, threshold):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df)

    distances = pairwise_distances(df, kmeans.cluster_centers_)
    min_distances = distances.min(axis=1)

    clusters = kmeans.labels_
    clusters[min_distances > threshold] = -1  

    return clusters

# Функция для DBSCAN
def dbscan_clustering(df, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df)
    return clusters

# Функция для иерархической кластеризации
def hierarchical_clustering(df, num_clusters, linkage, metric):
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage, metric=metric)
    clusters = hierarchical.fit_predict(df)
    return clusters

# Функция для OPTICS
def optics_clustering(df, min_samples, xi):
    optics = OPTICS(min_samples=min_samples, xi=xi)
    clusters = optics.fit_predict(df)
    return clusters

# Функция для масштабирования
def scale(df):
    # Выбор скейлера
    scaler_methods = {
        "-": None,
        "Standard Scaler": StandardScaler(),
        "Min-Max Scaler": MinMaxScaler(),
        "Robust Scaler": RobustScaler(),
        "Max Absolute Scaler": MaxAbsScaler()
    }

    selected_scaler = st.sidebar.selectbox("Выберите метод масштабирования", list(scaler_methods.keys()))

    # Применение выбранного метода масштабирования
    if selected_scaler != "-":
        scaler = scaler_methods[selected_scaler]
        scaled_data = scaler.fit_transform(df)
        st.write("##### Масштабированные данные")
        st.write(scaled_data)
        return scaled_data
    else:
        return df

# Функция для построения графика с кластерами (двумерные данные)
def plot_clusters(df, clusters):
    # Создать цветовую палитру для кластеров
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    cluster_colors = [colors[i % len(colors)] for i in clusters]

    # Визуализация данных с цветовым отображением кластеров
    fig, ax = plt.subplots()
    ax.scatter(df[:, 0], df[:, 1], c=cluster_colors)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Создание легенды
    legend_labels = list(set(clusters))
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i % len(colors)], markersize=10) for i in legend_labels]
    ax.legend(legend_handles, legend_labels, loc='upper right')

    st.pyplot(fig)


# Функция для построени графика силуэта (с аномалиями)
def plot_silhouette(df, clusters):

    # Вычисление коэффициента силуэта для каждого объекта
    silhouette_values = silhouette_samples(df, clusters)

    # Вычисление среднего коэффициента силуэта
    silhouette_avg = silhouette_score(df, clusters)

    fig_silhouette, axes = plt.subplots()

    # График силуэта
    y_lower = 10
    for i in range(np.min(clusters), np.max(clusters) + 1):
        # Выбор силуэтных значений для текущего кластера
        cluster_silhouette_values = silhouette_values[clusters == i]
        cluster_silhouette_values.sort()

        # Вычисление границы для текущего кластера на графике
        y_upper = y_lower + len(cluster_silhouette_values)

        if i == -1:
            # Обработка аномалий (кластер -1)
            color = 'gray'
            axes.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, alpha=0.7)
        else:
            # Обработка обычных кластеров
            color = plt.cm.get_cmap("Spectral")(i / (np.max(clusters) + 1))
            axes.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, alpha=0.7)

        # Настройка границы и метки для текущего кластера
        axes.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_values), str(i))
        y_lower = y_upper + 10

    # Оформление графика силуэта
    axes.set_xlabel("Значение коэффициента силуэта")
    axes.set_ylabel("Номер кластера")
    axes.axvline(x=silhouette_avg, color="red", linestyle="--")

    # Установка пределов осей Y для полного отображения графика
    axes.set_ylim([0, y_lower])

    st.pyplot(fig_silhouette)

    # Вывести значения элементов с коэффициентами силуэта ниже 0 на экран
    if any(value < 0 for value in silhouette_values):  
        st.write("**Элементы с коэффициентами силуэта ниже 0:**")
        sorted_indices = np.argsort(silhouette_values)  # Сортировка индексов по значению коэффициента силуэта
        for i in sorted_indices:
            value = silhouette_values[i]
            if value < 0:
                st.write(f"Элемент {i+1}: значение = {df[i]}  коэффициент силуэта = {value}")
    else:
        st.write("Все элементы имеют коэффициент силуэта больше или равный 0.")

        # Вывести средний коэффициент силуэта на экран
    st.write("Средний коэффициент силуэта:", silhouette_avg)

# Функция для кластеризации
def cluster(data):

    # Меню для выбора метода кластеризации  
    clustering_algorithm = st.sidebar.selectbox("Выберите алгоритм кластеризации", ("-", "K-means", "K-means with outliers", "Hierarchical", "DBSCAN", "Ensemble"))

    clusters = None

    # Регулирование параметров алгоритмов кластеризации
    if clustering_algorithm == "K-means":
        n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
        clusters = kmeans_clustering(data, n_clusters)
    if clustering_algorithm == "K-means with outliers":
        n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
        threshold = st.sidebar.slider("Пороговое значение", 0.1, 10.0, 1.7, step=0.1)
        clusters = kmeans_with_ouliers(data, n_clusters, threshold)
    elif clustering_algorithm == "Hierarchical":
        n_clusters = st.sidebar.slider("Количество кластеров", 2, 10, 3)
        linkage = st.sidebar.selectbox("Метод объединения", ("ward", "complete", "average", "single")) # If linkage is “ward”, only “euclidean” is accepted. 
        if linkage != "ward":
            metric = st.sidebar.selectbox("Метрика для расстояния", ("euclidean", "manhattan", "cosine"))
        else:
            metric = "euclidean"
            st.sidebar.info('При методе ward доступна только метрика euclidean', icon="ℹ️")
        clusters = hierarchical_clustering(data, n_clusters, linkage, metric)
    elif clustering_algorithm == "DBSCAN":
        eps = st.sidebar.slider("Максимальное расстояние", 0.1, 3.0, 0.5, step=0.1)
        min_samples = st.sidebar.slider("Минимальное количество соседей", 2, 10, 5)
        clusters = dbscan_clustering(data, eps, min_samples)
    elif clustering_algorithm == "OPTICS":
        min_samples = st.sidebar.slider("Минимальное количество соседей", 2, 10, 5)
        xi = st.sidebar.slider("xi", 0.01, 1.0, 0.05, step=0.01)
        clusters = optics_clustering(data, min_samples, xi)
    elif clustering_algorithm == "Ensemble":
        st.sidebar.write("**Параметры для DBSCAN**")
        eps = st.sidebar.slider("Максимальное расстояние", 0.1, 3.0, 0.5, step=0.1)
        min_samples = st.sidebar.slider("Минимальное количество соседей", 2, 10, 5)
        dbscan_labels = dbscan_clustering(data, eps, min_samples)

        st.sidebar.divider()

        st.sidebar.write("**Параметры для LocalOutlierFactor**")
        n_neighbors = st.sidebar.slider("Количество соседей", 5, 100, 20, step=1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof_scores = lof.fit_predict(data)

        st.sidebar.divider()

        st.sidebar.write("**Параметры для OneClassSVM**")
        kernel = st.sidebar.selectbox("Функция ядра", ("rbf", "linear"))
        nu =  st.sidebar.slider("Доля аномалий", 0.01, 0.1, 0.05, step=0.01)
        svm = OneClassSVM(kernel=kernel, nu=nu)
        svm_scores = svm.fit_predict(data)

        st.write("##### Отдельные результаты")

        # Строим графики отдельных результатов
        fig_sep, (ax1_sep, ax2_sep, ax3_sep) = plt.subplots(1, 3, figsize=(15, 5))

        ax1_sep.scatter(data.values[:, 0], data.values[:, 1], c=dbscan_labels)
        ax1_sep.set_xlabel('Feature 1')
        ax1_sep.set_ylabel('Feature 2')
        ax1_sep.set_title('DBSCAN')

        ax2_sep.scatter(data.values[:, 0], data.values[:, 1], c=lof_scores)
        ax2_sep.set_xlabel('Feature 1')
        ax2_sep.set_ylabel('Feature 2')
        ax2_sep.set_title('LocalOutlierFactor')

        ax3_sep.scatter(data.values[:, 0], data.values[:, 1], c=svm_scores)
        ax3_sep.set_xlabel('Feature 1')
        ax3_sep.set_ylabel('Feature 2')
        ax3_sep.set_title('OneClassSVM')

        plt.tight_layout()
        st.pyplot(fig_sep)
        
        st.write("##### Соединение результатов")

        # Соединение результатов
        any_outliers_labels  = np.minimum.reduce([dbscan_labels, lof_scores, svm_scores])
        common_outliers_labels  = np.maximum.reduce([dbscan_labels, lof_scores, svm_scores])

        any_outliers_labels [any_outliers_labels  != -1] = 0
        common_outliers_labels [common_outliers_labels  != -1] = 0

        # Строим графики общих результатов
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.scatter(data.values[:, 0], data.values[:, 1], c=any_outliers_labels)
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('Все аномалии')

        ax2.scatter(data.values[:, 0], data.values[:, 1], c=common_outliers_labels)
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.set_title('Согласованные аномалии')

        plt.tight_layout()
        st.pyplot(fig)
    
    # Вывести результат работы алгоритма
    if clusters is not None:
        st.write("##### Результаты кластеризации")

        st.write(clusters)
        plot_clusters(data.values, clusters)  # здесь было data

        if len(np.unique(clusters)) > 1:
            # Вызов функции plot_silhouette
            if st.button('Построить график силуэта'):
                plot_silhouette(data.values, clusters)  # здесь было data
                
                if -1 in clusters:
                    st.write("**Потенциальные аномалии:**")
                    for i, cluster_label in enumerate(clusters):
                        if cluster_label == -1:
                            st.write(f"Элемент: {i}, Значение: {data.values[i]}, Метка: {cluster_label}")
                            

# Функция для обработки категориальных признаков 
def encode(df):
    # Обработка категориальных признаков с помощью One-Hot Encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
    else:
        df_encoded = df

    # Отобразить обновленные данные в главной части экрана
        st.markdown("##### Обработанные данные")
        st.write(df_encoded)

# Функция для метода снижения размерности PCA
def run_PCA(df):
    # Применить PCA для снижения размерности до двух компонент
    if len(df.columns) > 2:
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df)
    else:
        df_pca = df.values
    return df_pca


def intro():

    st.write("# DataClust ✨")
    st.sidebar.success("Выберите действие")

    st.markdown(
        """
        Данный инструмент был создан с целью изучения алгоритмов кластеризации и их применения при решении задачи обнаружения аномалий в данных.
    """
    )

    st.divider()
    st.markdown("##### Инструкция к использованию")
    st.markdown("- Можно загрузить демонстрационный датасет с помощью пункта 'Загрузить пример'. Он позволит ознакомиться со всем реализованным в инструменте функционалом.")
    st.markdown("- Пункт 'Загрузить датасет' позволяет использовать произвольный датасет, но он должен быть в формате **csv-файла** и в нем **должен отсутствовать столбец с целевыми значениями**. Это можно сделать самостоятельно, например, загрузив файл в Jupyter Notebook и удалив столбец с метками классов с помощью Pandas функции (df.drop('название_столбца', axis=1))")


def first_example():
    st.write("## Исследование примера 🌠")

    df = pd.read_csv('data.csv')

    # Отображение данных и результат кластеризации
    if df is not None:
        # Отобразить данные в главной части экрана
        st.write("##### Исходные данные")
        st.write(df)

        scale(df)

        # Вызов функции выбора алгоритма кластеризации
        cluster(df)


def new_dataset():
    #st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write("## Исследование нового датасета 🌌")

     # Заголовок и боковая панель
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV-файл", type="csv")

     # Отображение данных и результат кластеризации
    if uploaded_file is not None:
        df_new = load_csv(uploaded_file)

        # Отобразить данные в главной части экрана
        st.write("##### Исходные данные")
        st.write(df_new)
            
        if df_new.select_dtypes(include=["object"]).shape[1] > 0:
            # Обработка категориальных признаков с помощью One-Hot Encoding
            categorical_columns = df_new.select_dtypes(include=['object']).columns
            df_encoded = pd.get_dummies(df_new, columns=categorical_columns)
        
            st.write("В наборе данных присутствуют категориальные признаки, которые необходимо сначала преобразовать")
            # Отобразить обновленные данные в главной части экрана
            st.write("##### Преобразованнные данные")
            st.write(df_encoded)
        else:
            df_encoded = df_new

        scaled_data = scale(df_encoded.values)

        # Применить PCA для снижения размерности до двух компонент
        if len(df_encoded.columns) > 2:
            st.write("Для проведения кластеризации необходимо уменьшить размерность набора данных")
            pca = PCA(n_components=2)
            df_pca = pd.DataFrame(pca.fit_transform(scaled_data), columns=['Component 1', 'Component 2'])
        else:
            df_pca = scaled_data
        
        cluster(df_pca)


page_names_to_funcs = {
    "Инструкция": intro,
    "Загрузить пример": first_example,
    "Загрузить датасет": new_dataset
}

def main():

    demo_name = st.sidebar.selectbox(" ", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()

if __name__ == "__main__":
    main()
