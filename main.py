{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMuUZvcjaqRzx0EZkLSWudk",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/hrishitapanjetha/Demographic-Data-Analyzer/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 392
        },
        "id": "dg_5czQAW-X4",
        "outputId": "d0144997-a8c9-48f3-9747-efe9d5fe38ad"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   age         workclass  fnlwgt  education  education-num  \\\n",
              "0   39         State-gov   77516  Bachelors             13   \n",
              "1   50  Self-emp-not-inc   83311  Bachelors             13   \n",
              "2   38           Private  215646    HS-grad              9   \n",
              "3   53           Private  234721       11th              7   \n",
              "4   28           Private  338409  Bachelors             13   \n",
              "\n",
              "       marital-status         occupation   relationship   race     sex  \\\n",
              "0       Never-married       Adm-clerical  Not-in-family  White    Male   \n",
              "1  Married-civ-spouse    Exec-managerial        Husband  White    Male   \n",
              "2            Divorced  Handlers-cleaners  Not-in-family  White    Male   \n",
              "3  Married-civ-spouse  Handlers-cleaners        Husband  Black    Male   \n",
              "4  Married-civ-spouse     Prof-specialty           Wife  Black  Female   \n",
              "\n",
              "   capital-gain  capital-loss  hours-per-week native-country salary  \n",
              "0          2174             0              40  United-States  <=50K  \n",
              "1             0             0              13  United-States  <=50K  \n",
              "2             0             0              40  United-States  <=50K  \n",
              "3             0             0              40  United-States  <=50K  \n",
              "4             0             0              40           Cuba  <=50K  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-4babe39f-8f1a-4931-8aa6-4c2e9d01cbc1\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>workclass</th>\n",
              "      <th>fnlwgt</th>\n",
              "      <th>education</th>\n",
              "      <th>education-num</th>\n",
              "      <th>marital-status</th>\n",
              "      <th>occupation</th>\n",
              "      <th>relationship</th>\n",
              "      <th>race</th>\n",
              "      <th>sex</th>\n",
              "      <th>capital-gain</th>\n",
              "      <th>capital-loss</th>\n",
              "      <th>hours-per-week</th>\n",
              "      <th>native-country</th>\n",
              "      <th>salary</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>39</td>\n",
              "      <td>State-gov</td>\n",
              "      <td>77516</td>\n",
              "      <td>Bachelors</td>\n",
              "      <td>13</td>\n",
              "      <td>Never-married</td>\n",
              "      <td>Adm-clerical</td>\n",
              "      <td>Not-in-family</td>\n",
              "      <td>White</td>\n",
              "      <td>Male</td>\n",
              "      <td>2174</td>\n",
              "      <td>0</td>\n",
              "      <td>40</td>\n",
              "      <td>United-States</td>\n",
              "      <td>&lt;=50K</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>50</td>\n",
              "      <td>Self-emp-not-inc</td>\n",
              "      <td>83311</td>\n",
              "      <td>Bachelors</td>\n",
              "      <td>13</td>\n",
              "      <td>Married-civ-spouse</td>\n",
              "      <td>Exec-managerial</td>\n",
              "      <td>Husband</td>\n",
              "      <td>White</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>13</td>\n",
              "      <td>United-States</td>\n",
              "      <td>&lt;=50K</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>38</td>\n",
              "      <td>Private</td>\n",
              "      <td>215646</td>\n",
              "      <td>HS-grad</td>\n",
              "      <td>9</td>\n",
              "      <td>Divorced</td>\n",
              "      <td>Handlers-cleaners</td>\n",
              "      <td>Not-in-family</td>\n",
              "      <td>White</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>40</td>\n",
              "      <td>United-States</td>\n",
              "      <td>&lt;=50K</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>53</td>\n",
              "      <td>Private</td>\n",
              "      <td>234721</td>\n",
              "      <td>11th</td>\n",
              "      <td>7</td>\n",
              "      <td>Married-civ-spouse</td>\n",
              "      <td>Handlers-cleaners</td>\n",
              "      <td>Husband</td>\n",
              "      <td>Black</td>\n",
              "      <td>Male</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>40</td>\n",
              "      <td>United-States</td>\n",
              "      <td>&lt;=50K</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>28</td>\n",
              "      <td>Private</td>\n",
              "      <td>338409</td>\n",
              "      <td>Bachelors</td>\n",
              "      <td>13</td>\n",
              "      <td>Married-civ-spouse</td>\n",
              "      <td>Prof-specialty</td>\n",
              "      <td>Wife</td>\n",
              "      <td>Black</td>\n",
              "      <td>Female</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>40</td>\n",
              "      <td>Cuba</td>\n",
              "      <td>&lt;=50K</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-4babe39f-8f1a-4931-8aa6-4c2e9d01cbc1')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-4babe39f-8f1a-4931-8aa6-4c2e9d01cbc1 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-4babe39f-8f1a-4931-8aa6-4c2e9d01cbc1');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ],
      "source": [
        "# Import of needed library\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import plotly.express as px\n",
        "\n",
        "# Import the data \n",
        "url = 'https://raw.githubusercontent.com/JakubPyt/Demographic_Data_Analyzer/main/adult.data.csv'\n",
        "data = pd.read_csv(url, sep=\";\")\n",
        "\n",
        "# Display first five rows of dataset\n",
        "data.head()"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def increase_font():\n",
        "  from IPython.display import Javascript\n",
        "  display(Javascript('''\n",
        "  for (rule of document.styleSheets[0].cssRules){\n",
        "    if (rule.selectorText=='body') {\n",
        "      rule.style.fontSize = '16px'\n",
        "      break\n",
        "    }\n",
        "  }\n",
        "  '''))\n",
        "\n",
        "get_ipython().events.register('pre_run_cell', increase_font)"
      ],
      "metadata": {
        "id": "Zx0lBuySXFN1"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Simply value_counts for column 'race'\n",
        "race_count = data.race.value_counts()\n",
        "print(race_count) "
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 139
        },
        "id": "8bPsacAHXWkf",
        "outputId": "28550966-a1b2-44e8-f909-75805d8a8a7b"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "White                 27816\n",
            "Black                  3124\n",
            "Asian-Pac-Islander     1039\n",
            "Amer-Indian-Eskimo      311\n",
            "Other                   271\n",
            "Name: race, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Matplotlib\n",
        "\n",
        "# Size of figure\n",
        "plt.figure(figsize=(18,8))\n",
        "\n",
        "# Create plot\n",
        "data['race'].value_counts().plot(\n",
        "    kind='bar',\n",
        "    color='darkblue'\n",
        ")\n",
        "\n",
        "# Chart settings\n",
        "plt.grid(True)\n",
        "plt.title('How many of each race are represented in this dataset?', fontsize=18)\n",
        "plt.xticks(rotation = 0, fontsize=12)\n",
        "plt.xlabel('Race', fontsize=14)\n",
        "plt.ylabel('Count', fontsize=14)\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 504
        },
        "id": "iX2SaysFYR4a",
        "outputId": "3d7dd07b-6bbb-48a0-e136-6d83213979b3"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1800x800 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABdUAAALLCAYAAAAfVGsDAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACGfklEQVR4nOzdeXgV5fk/4CdASNgSNgmyL1r3faUugCKItBbrrlVU1GrBCrRKaVFBrVbrWje0FdDWpe5tcQMRVAR3qVu1VUHcgigCyhpgfn/4PedHSAKTQEzQ+74uLsjMOzPPzJl5D/mcOe/kJEmSBAAAAAAAsF51aroAAAAAAADYVAjVAQAAAAAgJaE6AAAAAACkJFQHAAAAAICUhOoAAAAAAJCSUB0AAAAAAFISqgMAAAAAQEpCdQAAAAAASEmoDgAAAAAAKQnVAQCqYNy4cdGtW7coKCiInJycyMnJiWuuuaamy6oW48ePj5ycnOjUqVNNlwJ8x0ydOjXbh1bWptA3derUKXJycmL8+PE1snx16NGjR+Tk5MSoUaNquhQAqDFCdQA2eaNGjUr9C/ns2bOzbWvTL6hsWq688so45ZRT4rnnnoulS5dGq1atoqioKBo1alTTpQGkMnXq1Bg1apT3wip66KGHYtSoUfHQQw/VdCnfGePHj49Ro0bF1KlTa7qUjWb27NkxatSo9X4AsWzZshg/fnwceuih0bZt26hfv34UFhbGPvvsE7feemskSfLtFAxAavVqugAAgE3NH//4x4iI+OUvfxlXXHFF5Obm1nBFAJUzderUGD16dHTv3j1OOumkmi6nSgoLC2OrrbaKtm3bfuvbfuihh+K2226LAQMGRP/+/attO127do38/PwoLCystm3UFuPHj4+nnnoqIr65G/67YPbs2TF69OiIiHUG6126dIlPP/00IiLq1asXDRs2jEWLFsX06dNj+vTpMWHChLj//vujTh33RQLUFnpkAIBKmDdvXsydOzciIk477TSBOkANOeyww+Ltt9+OyZMn13Qp1Wby5Mnx9ttvx2GHHVbTpVCNiouL44gjjognn3wylixZEgsXLowPP/wwDj/88Ij45kOc22+/vYarBGBNQnUAgEpYsmRJ9t+NGzeuwUoAgO+CadOmxb333hs9e/bMfljfrl27uPPOO2OzzTaLiIgJEybUZIkArEWoDgDleOCBB+JHP/pRFBUVRf369aOoqCh+9KMfxYMPPlhu+x//+MeRk5MTv/71r8vM+/TTT7PjuO++++7lLr/VVltFTk5O3HrrralrXHN8+NmzZ8cHH3wQp512WnTo0CHy8/Oja9euMXLkyFi8eHF2mTfeeCN+9rOfRfv27SM/Pz+23HLLuPjii6OkpKTcbXz55Zdx6623xlFHHRU77LBDNG/ePPLz86Njx45x3HHHxXPPPVdhfZmx7jNf4Z48eXL069cvNttss8jPz49tttkmRo8eHcuWLSu13KpVq6Jdu3aRk5MTl19++TqPwa233ho5OTnRpEmT+Oqrr1Ieuf+vMq9z5mF6az4Qr3PnztnXoCoPyps9e3YMGTIktttuu2jcuHE0bNgwtt566zj77LNjzpw55S6zevXqmDx5cvzyl7+MvffeO9q1axf169ePFi1aRPfu3WPMmDEVvp5rmjhxYhxzzDHRsWPHaNCgQTRv3jx23HHHOOuss2LGjBnrXPbll1+Oo446KjbffPPIy8uLLl26xLBhw+LLL7+s9DGIKHuu3H///dG7d+9o1apV1KlTp9RX5t94440YNWpUHHDAAdG1a9do0KBBFBQUxC677BIjR46Mzz//fL3b+89//hODBg2KbbfdNpo0aRKNGzeOrbbaKo455pi4//77Y/Xq1eUu9/DDD8fhhx8ebdu2jby8vGjWrFnsv//+cdNNN8WKFSuqtO8b8xpb13GL+OZbFiNHjoxddtklCgsLIz8/P7p06RIDBw6MN998s0r1r/2QyVdffTWOP/74aNeuXeTm5pYZwmHFihVx4403Rs+ePaNly5ZRv379aN26dfzkJz+JRx99tMLtZLYxderUKC4ujsGDB0fnzp0jPz8/WrduHccff3y8/fbb5S67dl/53nvvxemnnx6dO3eOvLy8Mtfu6tWr44477ohDDjkk2zdsttlm0bt377jrrrsqHNt45cqVccstt0SPHj2iZcuWkZubGy1atIitttoqjj766HX271XpC9Z+SGfa6zJzPDJDUjz11FPZ47OuZ45UpcaMt99+O44//vho3bp19rw766yzst/6qap1Pai0qu9B65M552+77baIiLjtttvKHL+KxgRfsWJF/PGPf4yddtopGjVqFIWFhXHAAQfEY489VuH21vWg0qVLl8YVV1wR3bp1i2bNmkVubm5sttlmse2228aAAQPi/vvvr9S+ZaxatSquu+662HXXXaNRo0bRvHnz6NGjR9x3333rXXbWrFlx2WWXxcEHHxw/+MEPolGjRtG4cePYdtttY8iQIeWeK5nXMTP0y+jRo8sc09mzZ2/QNtb097//Pfr27RtFRUWRm5sbTZs2jS233DIOPfTQuOGGGyo8Jyrbh3bq1Cl69uyZ/XntfVpz2KUf/vCH5W6zfv360aFDh4iISp+rAFSzBAA2cRdccEESEUmat7VZs2Zl244bN67M/OXLlydHH310tk2dOnWSZs2aJXXq1MlOO/bYY5MVK1aUWu7KK69MIiLZZZddyqzzb3/7W6n1ffnll6Xmf/TRR9n577//fur9XnNf7r///qRp06ZJRCQFBQVJ3bp1s/P222+/ZMWKFcmECROShg0bJhGRFBYWJjk5Odk2Rx99dLnbWPPY1q1bN2nWrFmSl5eXnZaTk5Nce+2161y2e/fuyeWXX57k5OQkOTk5SdOmTUttu2fPnsnKlSvLXXbLLbdMVq9eXeEx2GuvvZKISE477bTUxy1JqvY6P/vss0lRUVHSsmXLbJuWLVsmRUVFSVFRUbL77rtXqoa//e1vpY5lXl5e0qBBg+zPTZo0SR5//PEyy635ukdE0rhx46SwsLDUtP322y9ZsmRJudtdvHhxcuSRR5Zq36RJk1Lr2GmnnUotM27cuCQiko4dOyZ33HFHkpubmz2P1jxm2223XfLVV19V6jgkSelzZdiwYdlzq1mzZkndunWTCy64INu2Y8eO2e3l5+cnzZs3L3U+tW3bNnn77bcr3NYf/vCHUjVn1rHmtLWv0SVLliRHHHFEqWNWUFBQart77713Mn/+/Crv+4ZeY+s7bpMmTcr2ERGR5ObmJo0aNcr+XL9+/eS2226rdP1TpkzJruO+++7LnhsFBQVJfn5+0r1792zb2bNnJ9ttt12pfVv73D3jjDPK3U5m/tixY5PWrVsnEZE0aNAgady4canX8tFHHy2z7JrXzB133JFdpmHDhkmjRo2Sjh07Ztt+8cUXyf7771+qprVrPPTQQ5Ply5eX2sbKlSuTgw46qMxya76WFb1HVbUvqOp1OWfOnKSoqCj7+ufm5mb7scyfu+++e6PUmCRJ8uijj5ZatnHjxkl+fn4SEcnmm2+ejB07NvV7+LqOwdo25D1oXTLvBZl9yM/PL3P8nn322Wz7TJ913XXXZd+zcnNzS527OTk5ya233lru9jLLr/1/lkWLFiU77bRTqXU0bdo0qVevXnZaecdlfZYtW5b06dOn1Pvjmsds+PDhSffu3ZOIKNXHZGTmZfqVFi1alDofCwsLk2eeeabUMnfffXdSVFSUPYcbNWpU5pjOmTNng7aRcfLJJ5d5D8383yjzZ9asWWWWq0ofuvvuuyfNmjXLtll7n375y1+u9/V48803s/+nu/zyy9fbHoBvj1AdgE3exgzVf/WrX2V/OT3vvPOy4dr8+fOT3/72t9llhw8fXmq5V155JfvL5xdffFFq3sCBA7MhU0QkDz74YKn5f/3rX6v0y++a+9K0adPkwAMPTN58880kSb4JAf/0pz9lfxEbOXJkUlhYmBx99NHJ7NmzkyRJkq+++ir53e9+l13HpEmTymzj5ptvTi644ILkpZdeyoZIq1evTt5///3k7LPPTnJycpK6desmr7zySpllM69L06ZNkzp16iQjRoxI5s2blyRJkixcuDA5//zzs9teO0z46KOPsrU/+eST5e7/a6+9ll3+pZdeqtSxq+rrnCSlj3t5v3inMXHixKROnTpJvXr1knPPPTeZNWtWsnr16mT16tXJ22+/nQ29CwoKkg8++KDUsh9++GFy/PHHJ//85z9LnWtfffVVMm7cuKRNmzZJRCRDhw4td9tHHXVU9lwdPnx48uGHH2bnzZs3L7njjjvKBJuZ4Kphw4ZJXl5ecuqpp2YDjsWLFyfXX399Ngw577zzKn08MudKJmQaPnx48tlnnyVJ8k3AkzlnkyRJTjzxxGT8+PGljsvy5cuTJ554Itlzzz2TiEh23XXXcrdz4403lgpGX3311ey8xYsXJxMnTkyOPvroZOHChaWW+9nPfpZERNKlS5fkjjvuyM5funRp8o9//CPp0qVLEhFJ//79K73vG+MaW99xe+2117IB6GmnnZa89dZb2RDxgw8+SH7xi18kEZHUq1cvefHFFytV/5qheuPGjZNDDjkk+c9//pOd/9///jdJkiT5+uuvk6233jqJiKRHjx7J1KlTk2XLliVJkiQLFixIrrrqqux+XHPNNWW2s2ZY1qFDh2TixInZD9yef/75ZIcddsheM2ue00lS+ppt3Lhxstdee5Xaz3feeSdJkm+C8UxYt/POOyf/+te/ksWLF2frv+2225JWrVolEZEMGTKk1DYy/Xh+fn7yl7/8JRtir169Opk7d27ywAMPJEcccUSZ/dqQvmBDr8s1Q+d12dD+KvPet+OOOybPP/98kiRJsmrVquTRRx9N2rVrVyqorKw0oXpV3oPSGDBgQBIRyYABA9bZLhOKN2vWLGnbtm3y0EMPZT+wffvtt5O99947e24uWLCgwuXX/j/LRRddlERE0rx58+T+++/PXk+rVq1KPv744+T222+v9AfOSZIkQ4cOzb4/Xnzxxdn+bu7cucmZZ56ZvQ4rCtXPPvvs5IYbbkj++9//JqtWrUqSJElKSkqS559/Pjn44IOTiEjatGlT7ge/6wrrN8Y2nnnmmez732WXXVbqPfTzzz9PHn/88WTAgAHJxx9/XGq5DelD1+wjK2v27NlJ586dk4hItt122wo/LAegZgjVAdjkrRmqr30X0Np/1rzLeO1fUD/66KPsHV4jRowod1uZu0Fzc3OTTz75JDt91apVSfPmzZOIb+4aX1PmF6LML/BnnXVWqfmZu6ZOOumkSu33mkHRdtttl/2Fek0nnHBCts1BBx1U7l3f++23XxIRycCBAyu1/SRJkkGDBlW47JqvS0W/IP/0pz9NIiLp1atXmXn9+/dPIiI55phjyl128ODB6wxQK7Ihr3OSbHiovmrVqmTLLbdMIiK5+eabK2x36KGHJhGRnH322ZVa/4svvphEfHOn39KlS0vNe+KJJ7K133jjjanXmQmu1hUgZY7ZFltsUal6k6T0uTJs2LBKL5/x1VdfJUVFRUlElLlLcf78+UmTJk2y59S6vgGxpqeffjqJiKRVq1al7pRc04cffpi9Y3HNoH5jSHuNreu4HXDAAes835MkSX75y18mEZH85Cc/qVR9awZGe+65Z4V3/F544YXZAHftb/pkPPDAA0nEN98AKSkpKTUvs4369esnb731Vpll586dm+2Df/GLX5Sat+Y127Fjxwq/TXH77bcnEZFsvfXW5YabSZIkL730UpKTk5PUr18/mTt3bnZ6Jmw8/fTTy12uPBvaF2zodZkmVN/QGjPHpUWLFqWOV8brr7+eDf6rK1Sv6nvQ+lQ2VM/Lyyv1gVPGZ599lr3r/W9/+1uFy6/9f5a+ffsmEZFccsklla69Ih9//HH2/bGiD0iPPfbY9R7XiqxcuTLZcccdk4hI/vrXv5aZnzZUr+o2LrvssiQikt69e1dqnRvSh1Y1VH/33XeTDh06JBGRdO3atcyHhQDUPGOqA/CdMnfu3HX+Wdd4y/fff3+sXLky8vPz4ze/+U25bUaOHBl5eXlRUlJSamzROnXqRPfu3SMi4sknn8xO/+CDD2LWrFmx5ZZbxoknnlhmfkTElClTIiJKjbtZWUOHDo28vLwy0/v06ZP9929+85vsuMfltXnttdcqvd1+/fpFxDcP2KpIXl5euWPNR0T85Cc/qXDbZ555ZkREPPjgg2Vet6VLl8bf/va3iIj4+c9/XqmaN+R13hiefvrp+N///hctW7aMU089tcJ2mfPl8ccfr9T6d99992jVqlUsXrw4Zs6cWWre2LFjIyJi++23zx7fyho5cmS50zOv5bvvvlvqYa6VUadOnRg+fHiVlo345sGxmetw7XPyvvvui6+++ipyc3PjqquuKvdaKE9mHOzjjz8+2rdvX26bdu3aZa/fyr5e65PmGlvXcZs9e3Y8+eSTUa9evQqvw4j/f7498cQTsWrVqirVes4550TdunXLnZc5jsOGDcs+hG9t/fv3j4KCgvj888/j5ZdfLrfNkUceGdtss02Z6a1atYozzjgjIr4ZL7kigwcPrvABw5kazzzzzCgsLCy3zW677RbbbbddrFixItt3R0Q0bdo0IiKKi4sr3PbaNmZfUF3X5YbUmCRJ9rU444wzolWrVmWW23777eOII46odF2VUdX3oI3tiCOOiK233rrM9M022yy6detW6Toy59ynn366UeqL+KafXLlyZTRo0KDCY7b2sxoqo27dunHwwQdHxLr7tA2xrm1kjtm8efNS93PfZh+a8emnn8YBBxwQc+bMie233z6eeeaZaNeu3QatE4CNr15NFwAAG1NSwQPkMmbPnh2dO3cud95LL70UERF77LFHFBQUlNumWbNmsfvuu8ezzz6bbZ9xwAEHxIMPPlgqNM/8O/NQxQ4dOsSbb74Zn332WbRq1SpmzZqVffjWhoTqe+65Z7nTi4qKsv/eY4891tmmoodMvv/++3HjjTfGlClT4r333ouvvvqqzIMcP/roowpryzzUrjxt2rSJiIj58+eXmXfQQQdF165d47333ovbb789hg0blp133333xYIFC6Jx48Zx3HHHVbjt8mzo67yhnn322YiIWLhwYXb/y5N58OUHH3xQ7ryxY8fGAw88EG+88UZ88cUX5T4oc+3XZfr06RER8aMf/ahKtTdv3jy22GKLcuetuS9ffvllNGzYsNLr32KLLcoN3tY2YcKE+Otf/xovvvhizJ07t9ywsKJ932233WLzzTdPXVPm9br11lvjzjvvrLDdwoULI6L812t9NvQaW9dxy9S/evXq2HbbbStcRyYEWrx4cXzxxRepXoe17bPPPuVO//jjj7PHZeDAgRUG7xERX3/9dUR8cxz32muvMvMPOOCACpc94IAD4pJLLokvvvgiZs2aVW5fX1GNq1atyj4UdtSoUXHJJZdUuJ1Mf7Xma33IIYfEH/7wh/jnP/8Zffv2jRNPPDG6d+++zmt8Y/QFEdV7XW5IjbNmzcoeq/W9bnfddVel6qqMqr4HbWzlnc8bUsePfvSjuOuuu+L666+PefPmxdFHHx377rtvtGzZsso1Zt7vdt999wrfH3/wgx9E27Zt4+OPP65wPc8880zceuut8dxzz8VHH31U6oHpGevq09KoyjYOPPDAyM/Pj1dffTX222+/GDhwYBxwwAEV/r8w4tvtQzPOOeecmDNnTrRt2zYmT568QesCoPoI1QHg/3z22WcREdG2bdt1tsvcLZRpn5EJxf/zn/9EcXFxtG7dOnsnYyZQ6NmzZ9x2223x5JNPxjHHHJOd37Vr1wrvgk2jSZMm5U6vV69e6jYlJSVl5j344INx7LHHxvLly7PTCgoKIj8/P3JycmLFihXx5ZdflvvL7Pq2u+a2V65cWWZeTk5OnH766TF8+PD485//XCpUv+WWWyIi4rjjjqswLKnIhr7OG+qTTz6JiG+O99y5c9fbfunSpaV+/uyzz6JXr17x+uuvZ6fl5+dHy5Yts2HlvHnzYvXq1WVel8xdtB07dqxS7Wley4jyz6U01hccrF69On72s5+VCuDq1asXzZo1i/r160fEN+HfsmXLNtq+Z16vRYsWxaJFi9bbvrJ3A2+Ma2xdxy1T/+rVq1OdbxGV34f11ZGpISLW+W2hNDWs67pdc95nn31WblBWUY3z58/PvgYVfcC4rhr33XffuOyyy2LkyJHx2GOPxWOPPRYR3/QjvXr1ihNPPLHMB6cb2hdkVOd1uSE1rtl3rut1q+47cKv6HlQTdVTmNTruuOPihRdeiOuuuy7uvvvuuPvuuyPimw/ZevfuHaecckrstttulaqxMu+PFYXqw4cPj8svvzz7c926dUv10V9//XUsXrx4nX3a+lR1G127do2//OUvccYZZ8SMGTNixowZEfHNtwV69uwZxx13XBx66KGlvsn0bfahGZlvfJx55pkCdYBazPAvALCRbLfddtm7vjN3qE+ZMiVycnKyYUomXM/Mz/y9IXepV5cvvvgiTjrppFi+fHkccMABMXXq1FiyZEksXLgw5s6dG8XFxXHvvfdWaw2nnHJK5OXlxdtvvx1PP/10RES8/fbb2a90n3766dW6/eqQuZttr732iuSb59us98+ahg4dGq+//nq0aNEixo4dG59++mksXbo05s2bF8XFxVFcXJy963HtZdMOeVJT1nUHc8Q3d4vfddddUbdu3Tj//PPjf//7Xyxfvjzmz5+f3ffMUBIba98zr9dNN92U6rUaP3586nVvrGtsXcctU39RUVHq861Tp06p9yFNHWsOhfCf//wnVQ0nnXRSlWrYGDU++uijqWpcexiMc845J2bNmhVXX3119O/fP1q1ahUfffRRjB8/Pg444IA48sgjS4WmG9oXfBs2hRq/z6655pp455134pJLLom+fftG06ZN4913340bb7wxdt999xgyZMi3Ws+kSZOyYfcvfvGLeP3118v00UOHDo2I9X+zsLq2cfzxx8cHH3wQY8aMiaOPPjrat28f8+bNi3vuuSf69+8f3bt3L/UB6rfZh2Z88cUXERHr/HYIADVPqA4A/ydzN9D6vpKcmV/e3UM9evSIiG/C8v/+97/x0Ucfxfbbbx+bbbZZRPz/8HzN0D1i3V+NrymPPPJILFq0KJo1axb/+te/onv37tGgQYNSbSozfnBVtGzZMg4//PCIiPjzn/9c6u/ddtut0nfhRWyc13lDtG7dOiKqNkxISUlJPPDAAxERcf3118fJJ5+cXV/GqlWrKrwbeEO2XRtk7sQ89dRTY/To0bHFFltEnTql/ztb0TlZ1X2vzmP2bVxjmfo///zzDbozdGPUELHhx3FdQ06sOa+y122LFi2ydwtvSI1t2rSJIUOGxIMPPhhz586N1157LTsW+X333Rc33XRTtu2mcD1uSI1rvgZpXzcqb4sttogRI0bEI488El988UXMmDEj+vfvHxER1157bfzzn/9Mva7Ma7a+16Si+Zk+uk+fPnHDDTfE9ttvX+aDrA3t0zbGNpo3bx4///nP4+677445c+bEu+++m33uzDPPPFPqA7Oa6EMzHwbU9g/CAb7vhOoA8H923333iPhmTNHM+MhrW7BgQakxude2ZmheXmDevn372GKLLeK9996LSZMmZb9WnAnja5MPP/wwIiK22mqrCsfhfeKJJ6q9jswDNe+7774oLi6O22+/PSKqfpf6xnidN0RmTOfi4uJKj9c+b968WLZsWURE7LLLLuW2mTZtWrbN2n74wx9GRMS//vWvSm23tsickxXt+9dffx3PP/98ufMy+/7SSy9V6sF+mddrwoQJlSk1lW/jGsvUv2rVqnj00Uc3aF1V1alTp+xwEht67q35cNCK5jVv3nydYySXJzc3N/tsio15feywww7x5z//Ofs6TJo0KTtvQ/qCjSHzgdS67hjekBo7d+4czZs3j4h1v25rP7x7U5Hm+H3b6tSpE3vvvXfcd9990aFDh4gofc6tz5rvj5nnG6ztf//7X4UfSq+vj06SZJ2vd5pjuqHbKE/Xrl3j0ksvzT6jpbzrtKp96Jof/KY9V6ZMmRJTpkzJPnAVgNpJqA4A/+fwww+PevXqxbJly+Kyyy4rt80ll1wSy5cvj9zc3Owd1GvKBOizZs2KcePGlZqWkQnezzvvvIiI2HrrrSv14MRvS2FhYURE/Pe//y03pJ05c+Y6H9y4sey7776x/fbbx7Jly+Loo4+Ozz//vEoPKM3YGK/zhujZs2f2oYJDhw4t9wGja1rzwXUFBQXZO9f+/e9/l2m7cuXK+N3vflfhugYOHBgREW+++WapO2Y3FZlzsrx9j4i46KKL4quvvip33pFHHhkFBQWxcuXKGDp0aOpwI/PhzRtvvLHeY7Z48eL1vp5r+jausS233DL7od3vfve7Cj9IyqiuBzaedtppEfHNED6vvvpqlWu4995745133ikz/fPPP4+bb745IiKOPvroKtWYea0feeSReOSRRypV45pj4pcn8w2ENQO2DekLNobMgygXLFhQYZsNqTEnJyeOOuqoiIgYM2ZMud+geeutt+K+++6rbOm1QprjV53Wdc7VrVs3O7742t/mWZfDDz886tatG0uXLo0rrrii3DYXXnhhhcuvr48eM2ZMvP/++xUun+aYbsg2qnKdbmgfuuYDX9OeKz169IgePXqU+SYaALWLUB0A/k/btm3j7LPPjoiIP/zhD3HBBRdkfwFasGBBnHfeefHHP/4xIiKGDRtWbhC+5ZZbZh+69vzzz0fdunWje/fupdpkQvbMHbW1cTz1iIjevXtHnTp1Yv78+XH88cdnv+69YsWKuOeee6J3797rfPDaxvTzn/88IiI7rnpVHlCasTFe5w1Rr169GDNmTNSrVy+mTZsW+++/f0yePLnUWMvvv/9+jBkzJvbYY4+48cYbs9MbN26cvWtu2LBh8eSTT8bq1asj4pvQ95BDDomXXnopGjVqVO62e/bsGcccc0xERAwePDhGjBhR6o7Dzz//PP7yl79kw/faJnPX3p///Oe45ZZbsgFfZgzdyy+/PFq0aFHusoWFhdlxeP/+97/HYYcdFjNnzszOX7JkSTz88MPxk5/8pNR4ut27d4+TTz45IiIGDRoUQ4cOLRXYLF++PJ577rk499xzo2PHjpV6sO23dY1dd9110bhx4/jvf/8be++9d/zjH/8oFeJ//PHH8de//jUOPPDAGD58+AZvrzy/+tWvYocddohly5ZFz5494/rrr8+OGxzxzbX36KOPxoknnhj77bdfhevJz8+Pgw8+OJ544onsByMvvvhi9OrVKz7//PNo0qRJ/OY3v6lSjT/72c+iV69ekSRJHHbYYXHxxReXesjq4sWLY8qUKTFo0KDo0qVLqWX79+8fp5xySjz66KOlgrP58+fHxRdfHJMnT46IiH79+mXnbUhfsDFsv/32EfHNh2zTp08vt82G1jhixIho0qRJfP7553HQQQdl73ZPkiQmTpwYffv2rfBbGrVd5vg988wz8fbbb3/r299rr73il7/8ZUydOrXUsCSffPJJnHXWWfHuu+9GRMQhhxySep1t27aNQYMGRcQ3H1Jeeuml2Q8q582bF4MHD46//e1v2WB7bZk++tFHH42LLrooW9eCBQvikksuibPOOqvCPjri/x/TRx55pMIhZjZkG4MHD46jjjoq7r///lJ99ddffx1jxozJfhNuzes0YsP60B/84AfZDzj+8pe/pPpANycnJ3Jycso8twGAWiYBgE3cBRdckEREkuZtbdasWdm248aNKzN/+fLlyVFHHZVtU6dOnaRZs2ZJnTp1stOOPfbYZMWKFRVu44QTTsi23WOPPcrMLy4uzs6PiOSee+6p1P6Wty+zZs0qt82UKVPWe2zGjRuXRETSsWPHMvOGDx9eqtbCwsIkNzc3iYikc+fOyR133FHh+jOvS/fu3Svcdpr6kiRJFi5cmDRq1Cjb9qWXXlpn+/XZkNc5zXFP48EHH0yaNGmSXVdubm7SokWLJC8vr9Qxv/jii0st99JLL5U6Fnl5edn11KtXL7n99tuTjh07VniOL168OPnpT39aahsFBQVJYWFh9ueddtqp1DLrOkc2xnFJc64kSZJ8+eWXydZbb13qdWvatGmSk5OTRETy85//PBkwYEASEcmAAQPKXccll1xS6nVu0KBB0rx581LTvvzyy1LLLF++PDn11FNLHbPGjRuXOWciIvnoo48qte/VfY1lTJs2LWndunV2XXXr1k1atGiRNGjQoNT2Tz311ErVn/YaTpIk+fjjj5O999472z4nJydp2rRpUlBQUKqGLbbYosyymXljx47N7kfDhg2Txo0bl7oWJkyYUGbZypybCxcuTH70ox+VuT7WPM8y19qaunfvXmaZtffriCOOSFatWlVmm1XtCzb0uiwpKUm22mqr7PxmzZolHTt2TDp27Jjce++9G6XGJEmSCRMmlGrXpEmT7Hm3+eabJ2PHjk19Dq1tXcdgY74HlWf+/PnJZpttll2+ZcuW2eM3Y8aMbLt19ccZ6+q3Klo+M33Na2nN94aISIYOHVrp/Vq6dGnSq1evUn1Fs2bNsuf/8OHDs+f7BRdcUGrZFStWJPvtt1+putbsJ/v165eMHDmywtflv//9b5Kfn5/t34uKirLH9MMPP9zgbWSO85r9eNOmTUtN23fffZOvv/66TG0b0ocOHDgwO79hw4ZJhw4dko4dOya/+tWvyn0NMm3XPr4A1C7uVAeANdSvXz/+/ve/x3333Rd9+/aNFi1axFdffRUtWrSIvn37xgMPPBB33nln5ObmVriONe88L+8BpEVFRbHttttGxDd3I9XG8dQz/vCHP8Ttt98ee+65ZzRo0CBKSkpiiy22iN/+9rfx6quvRps2bb6VOgoKCqJ3794RUfUHlK5pY7zOG6p///7x7rvvxgUXXBB77rlnNG7cOBYsWBB5eXmx0047xamnnhoPPvhgnHPOOaWW22233eKFF16Io446Klq2bBmrV6+OJk2axFFHHRXTp0+PE044YZ3bbdiwYdx///0xYcKEOOyww6JNmzaxbNmyqFevXuy4447xy1/+Mm655ZZq2+8N0bRp05g+fXoMGTIkOnXqFHXr1o169epFjx494q677ooxY8asdx0jRoyIf//733Haaadlh7VYsWJFbLnllnHsscfGAw88UOrr+hHfnC9//vOfY/r06XHSSSdF165dY9WqVfH1119Hq1atokePHnH++efHa6+9lh07PK1v6xrbZ5994r///W9cccUVsf/++0fTpk1jwYIFUbdu3dhmm23iZz/7Wdxxxx1xzTXXbJTtladNmzYxbdq0uOuuu+LQQw+NzTffPJYsWRIrVqyITp06xY9//OO45pprst9IKU/nzp3j1VdfjUGDBsVmm20WK1asiFatWsWxxx4br776apk7TCuroKAg/vWvf8UjjzwSRx99dHTo0CGWL18eS5YsibZt20bv3r3j0ksvLTMEzXXXXReXXXZZHHLIIbHllltGkiSxdOnSaNOmTRx66KFx//33x7333lvuUBxV7Qs2VL169WLy5Mlx6qmnRufOnWPx4sXxwQcfxAcffFBmPO0NqbFfv37xyiuvxDHHHBOtWrWKFStWRFFRUQwePDheffXVSo9/X1s0a9Ysnn766TjmmGOibdu2sXDhwuzxq+i5FhvT3XffHaNHj44DDzwwOnfuHCtWrIiSkpLo2LFjHH300TF58uS46qqrKr3e/Pz8ePTRR+Paa6+NnXfeOerXrx9JksR+++0X99xzT/zhD3+ocNnc3NyYOHFiXHDBBfGDH/wgcnNzI0mS2HPPPeOmm26Kf/7zn2UeKrqmLbfcMqZMmRKHHnpobLbZZvHFF19kj+nKlSs3eBvnnXde/OlPf4rDDjsstt5666hXr162Hz/ooINi7NixMXXq1HK/7bUhfegNN9wQo0aNih122CEiIubMmRMffPBBhQ8VB2DTkJMktejJKgAA5Vi+fHm0bds2vvjii7j55pur/JBSYNOUeZbAlClTavUHkQAAfD+4Ux0AqPXuuuuu+OKLL6KgoKDKDygFAACAjUGoDgDUau+9916cd955ERFxxhlnVPkBpQAAALAx1KvpAgAAyrPvvvvGrFmzori4OFavXh3t2rWLESNG1HRZAAAAfM+5Ux0AqJU++uij+OSTT6JZs2Zx2GGHxZQpU6Jp06Y1XRYAAADfcx5UCgAAAAAAKblTHQAAAAAAUjKmejVavXp1fPLJJ9GkSZPIycmp6XIAAAAAAChHkiTx1VdfRZs2baJOnXXfiy5Ur0affPJJtG/fvqbLAAAAAAAghQ8//DDatWu3zjZC9WrUpEmTiPjmhSgoKKjhaqjtSkpKYuLEidG7d+/Izc2t6XKA7wh9C1Ad9C1AddC3ANVB30JaixYtivbt22cz3XURqlejzJAvBQUFQnXWq6SkJBo2bBgFBQU6eWCj0bcA1UHfAlQHfQtQHfQtVFaaYbw9qBQAAAAAAFISqgMAAAAAQEpCdQAAAAAASEmoDgAAAAAAKQnVAQAAAAAgJaE6AAAAAACkJFQHAAAAAICUhOoAAAAAAJCSUB0AAAAAAFISqgMAAAAAQEpCdQAAAAAASEmoDgAAAAAAKQnVAQAAAAAgJaE6AAAAAACkJFQHAAAAAICUhOoAAAAAAJCSUB0AAAAAAFISqgMAAAAAQEpCdQAAAAAASEmoDgAAAAAAKQnVAQAAAAAgJaE6AAAAAACkJFQHAAAAAICUhOoAAAAAAJCSUB0AAAAAAFISqgMAAAAAQEr1aroAvp9ycq6o6RJqnQYN6sRdd3WNwsLrYunS1TVdTq2RJL+u6RIAAAAAIMud6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKQkVAcAAAAAgJSE6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKQkVAcAAAAAgJSE6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKQkVAcAAAAAgJSE6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKQkVAcAAAAAgJSE6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKQkVAcAAAAAgJSE6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKQkVAcAAAAAgJSE6gAAAAAAkJJQHQAAAAAAUhKqAwAAAABASkJ1AAAAAABISagOAAAAAAApCdUBAAAAACAloToAAAAAAKRU60L1Sy+9NPbYY49o0qRJtGrVKvr37x/vvPNOqTY9evSInJycUn/OOOOMUm3mzJkT/fr1i4YNG0arVq3inHPOiZUrV5ZqM3Xq1Nh1110jLy8vtthiixg/fnyZem644Ybo1KlT5Ofnx1577RUvvPDCRt9nAAAAAAA2DbUuVH/qqadi0KBB8dxzz8WkSZOipKQkevfuHYsXLy7V7rTTTotPP/00++fyyy/Pzlu1alX069cvVqxYEdOnT4/bbrstxo8fH+eff362zaxZs6Jfv37Rs2fPmDlzZgwZMiROPfXUePzxx7Nt/v73v8ewYcPiggsuiFdeeSV22mmn6NOnT3z22WfVfyAAAAAAAKh16tV0AWt77LHHSv08fvz4aNWqVbz88sux//77Z6c3bNgwWrduXe46Jk6cGG+99VY88cQTUVRUFDvvvHNcdNFFMXz48Bg1alTUr18/xowZE507d44rr7wyIiK22WabmDZtWlx99dXRp0+fiIi46qqr4rTTTouTTz45IiLGjBkTDz/8cIwdOzZ+85vflNnu8uXLY/ny5dmfFy1aFBERJSUlUVJSsgFH5bunQYNa93lOjWvQIGeNvx2fDNcObJjMNeRaAjYmfQtQHfQtQHXQt5BWZc6RnCRJkmqsZYO9++67seWWW8brr78e22+/fUR8M/zLm2++GUmSROvWrePHP/5xnHfeedGwYcOIiDj//PPjn//8Z8ycOTO7nlmzZkWXLl3ilVdeiV122SX233//2HXXXeOaa67Jthk3blwMGTIkFi5cGCtWrIiGDRvGfffdF/3798+2GTBgQCxYsCD+8Y9/lKl11KhRMXr06DLT77zzzmxtAAAAAADULkuWLInjjjsuFi5cGAUFBetsW+vuVF/T6tWrY8iQIbHPPvtkA/WIiOOOOy46duwYbdq0iddeey2GDx8e77zzTjzwwAMREVFcXBxFRUWl1pX5ubi4eJ1tFi1aFEuXLo0vv/wyVq1aVW6bt99+u9x6R4wYEcOGDcv+vGjRomjfvn307t17vS/E901h4XU1XUKt06BBTowd2yVOOeX9WLq0Vn/W9a1auPCsmi4BNmklJSUxadKkOOiggyI3N7emywG+I/QtQHXQtwDVQd9CWplRR9Ko1aH6oEGD4o033ohp06aVmn766adn/73DDjvE5ptvHgceeGC899570bVr12+7zKy8vLzIy8srMz03N9dFu5alS1fXdAm10DdDvixdmjg+a3DtwMbhvQio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AICUhOoAAAAAAJBSlUP1p59+OubMmbPONh9++GE8/fTTVd0EAAAAAADUKlUO1Xv27Bnjx49fZ5vbb789evbsWdVNAAAAAABArVLlUD1JkvW2Wb16deTk5FR1EwAAAAAAUKtU65jq//vf/6KwsLA6NwEAAAAAAN+aepVpfMopp5T6+aGHHorZs2eXabdq1arseOp9+/bdoAIBAAAAAKC2qFSovuYY6jk5OTFz5syYOXNmuW1zcnJijz32iKuvvnpD6gMAAAAAgFqjUqH6rFmzIuKb8dS7dOkSQ4YMibPPPrtMu7p160azZs2iUaNGG6dKAAAAAACoBSoVqnfs2DH773HjxsUuu+xSahoAAAAAAHyXVSpUX9OAAQM2Zh0AAAAAAFDrVTlUz3jhhRfixRdfjAULFsSqVavKzM/JyYnzzjtvQzcDAAAAAAA1rsqh+vz586N///7x7LPPRpIkFbYTqgMAAAAA8F1R5VB92LBhMW3atOjRo0cMGDAg2rVrF/XqbfCN7wAAAAAAUGtVOQWfMGFC7LnnnjF58uTIycnZmDUBAAAAAECtVKeqCy5dujT233//jR6oP/300/HjH/842rRpEzk5OfHQQw+Vmn/SSSdFTk5OqT8HH3xwqTbz58+P448/PgoKCqJp06YxcODA+Prrr0u1ee2112K//faL/Pz8aN++fVx++eVlarn33ntj6623jvz8/Nhhhx3ikUce2aj7CgAAAADApqXKofrOO+8cs2fP3oilfGPx4sWx0047xQ033FBhm4MPPjg+/fTT7J+77rqr1Pzjjz8+3nzzzZg0aVJMmDAhnn766Tj99NOz8xctWhS9e/eOjh07xssvvxx//OMfY9SoUXHLLbdk20yfPj2OPfbYGDhwYLz66qvRv3//6N+/f7zxxhsbfZ8BAAAAANg0VHn4lwsuuCAOPfTQeO6552LvvffeaAX17ds3+vbtu842eXl50bp163Ln/ec//4nHHnssXnzxxdh9990jIuK6666LQw45JK644opo06ZN3HHHHbFixYoYO3Zs1K9fP7bbbruYOXNmXHXVVdnw/dprr42DDz44zjnnnIiIuOiii2LSpElx/fXXx5gxYzba/gIAAAAAsOmocqheXFwc/fr1i+7du8fxxx8fu+66axQUFJTb9sQTT6xygeWZOnVqtGrVKpo1axYHHHBAXHzxxdGiRYuIiJgxY0Y0bdo0G6hHRPTq1Svq1KkTzz//fBx22GExY8aM2H///aN+/frZNn369InLLrssvvzyy2jWrFnMmDEjhg0bVmq7ffr0KTMczZqWL18ey5cvz/68aNGiiIgoKSmJkpKSjbHr3xkNGlT5SxLfWQ0a5Kzxt+OT4dqBDZO5hlxLwMakbwGqg74FqA76FtKqzDlS5VA9M7Z5kiQxfvz4GD9+fJnx1ZMkiZycnI0aqh988MHx05/+NDp37hzvvfde/Pa3v42+ffvGjBkzom7dulFcXBytWrUqtUy9evWiefPmUVxcHBHffCDQuXPnUm2Kioqy85o1axbFxcXZaWu2yayjPJdeemmMHj26zPSJEydGw4YNq7S/31V33dW1pkuotcaO7VLTJdQqnmUAG8ekSZNqugTgO0jfAlQHfQtQHfQtrM+SJUtSt61yqD5u3LiqLrpBjjnmmOy/d9hhh9hxxx2ja9euMXXq1DjwwANrpKaMESNGlLq7fdGiRdG+ffvo3bt3hXfxf18VFl5X0yXUOg0a5MTYsV3ilFPej6VLk5oup9ZYuPCsmi4BNmklJSUxadKkOOiggyI3N7emywG+I/QtQHXQtwDVQd9CWplRR9Kocqg+YMCAqi66UXXp0iVatmwZ7777bhx44IHRunXr+Oyzz0q1WblyZcyfPz87Dnvr1q1j7ty5pdpkfl5fm4rGco/4Zqz3vLy8MtNzc3NdtGtZunR1TZdQC30z5MvSpYnjswbXDmwc3ouA6qBvAaqDvgWoDvoW1qcy58cmP3DzRx99FF988UVsvvnmERHRrVu3WLBgQbz88svZNk8++WSsXr069tprr2ybp59+utQ4OZMmTYqtttoqmjVrlm0zefLkUtuaNGlSdOvWrbp3CQAAAACAWqrKd6rPmTMnddsOHTqkbvv111/Hu+++m/151qxZMXPmzGjevHk0b948Ro8eHYcffni0bt063nvvvTj33HNjiy22iD59+kRExDbbbBMHH3xwnHbaaTFmzJgoKSmJwYMHxzHHHBNt2rSJiIjjjjsuRo8eHQMHDozhw4fHG2+8Eddee21cffXV2e2effbZ0b1797jyyiujX79+cffdd8dLL70Ut9xyS+p9AQAAAADgu6XKoXqnTp3KPJi0PDk5ObFy5crU633ppZeiZ8+e2Z8zY5QPGDAgbrrppnjttdfitttuiwULFkSbNm2id+/ecdFFF5UaduWOO+6IwYMHx4EHHhh16tSJww8/PP70pz9l5xcWFsbEiRNj0KBBsdtuu0XLli3j/PPPj9NPPz3b5oc//GHceeedMXLkyPjtb38bW265ZTz00EOx/fbbp94XAAAAAAC+W6ocqp944onlhuoLFy6Mf//73zFr1qzo3r17dOrUqVLr7dGjRyRJxQ9pfPzxx9e7jubNm8edd965zjY77rhjPPPMM+tsc+SRR8aRRx653u0BAAAAAPD9UOVQffz48RXOS5Ikrrzyyrj88svj1ltvreomAAAAAACgVqmWB5Xm5OTEr3/969huu+3inHPOqY5NAAAAAADAt65aQvWM3XffPZ588snq3AQAAAAAAHxrqjVUf++99yr1kFIAAAAAAKjNqjymekVWr14dH3/8cYwfPz7+8Y9/xIEHHrixNwEAAAAAADWiyqF6nTp1Iicnp8L5SZJEs2bN4sorr6zqJgAAAAAAoFapcqi+//77lxuq16lTJ5o1axZ77LFHnHzyydGqVasNKhAAAAAAAGqLKofqU6dO3YhlAAAAAABA7VetDyoFAAAAAIDvko3yoNJnn302Zs6cGYsWLYqCgoLYeeedY5999tkYqwYAAAAAgFpjg0L16dOnx8knnxzvvvtuRHzzcNLMOOtbbrlljBs3Lrp167bhVQIAAAAAQC1Q5VD9zTffjN69e8eSJUvioIMOip49e8bmm28excXFMWXKlJg4cWL06dMnnnvuudh22203Zs0AAAAAAFAjqhyqX3jhhbFixYp45JFH4uCDDy41b/jw4fHYY4/FoYceGhdeeGHcfffdG1woAAAAAADUtCo/qHTq1KlxxBFHlAnUMw4++OA44ogjYsqUKVUuDgAAAAAAapMqh+oLFy6Mzp07r7NN586dY+HChVXdBAAAAAAA1CpVDtXbtGkTzz333DrbPP/889GmTZuqbgIAAAAAAGqVKofqhx56aEydOjXOO++8WLZsWal5y5YtiwsuuCCmTJkSP/nJTza4SAAAAAAAqA2q/KDS8847LyZMmBCXXHJJ3HzzzbHnnntGUVFRzJ07N1588cWYN29edOnSJc4777yNWS8AAAAAANSYKofqLVq0iOeeey7OPffcuPvuu+ORRx7JzsvPz4+TTz45LrvssmjevPlGKRQAAAAAAGpalUP1iIiW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s2fPrv6igE1STk5OjBo1qlrWPXXq1MjJyYn77ruvWtYP3weZ62jq1Kk1XcomqTr7uPXp1KlTnHTSSTWybfiuGTVqVOTk5JSa9l25xnr06BHbb7/9ett9V/YXNjVp8xeoCqE6VMI999wTOTk58eCDD5aZt9NOO0VOTk5MmTKlzLwOHTrED3/4ww3e/o033hjjx4/f4PUAtVfmA7U1/7Rq1Sp69uwZjz76aE2XB98rN954Y+Tk5MRee+1V06VU2uzZs0v1I3Xr1o0OHTrEYYcdFjNnzvxWaznppJOicePG3+o2oaZtqv1H2pB4U9ajR48y/9fK/Nl6661rujzg/7z55pvxs5/9LNq2bRt5eXnRpk2bOP744+PNN98s1W769OkxatSoWLBgQc0UyvdWvZouADYl++67b0RETJs2LQ477LDs9EWLFsUbb7wR9erVi2effTZ69uyZnffhhx/Ghx9+GMccc0yltnXCCSfEMcccE3l5edlpN954Y7Rs2dJdDvA9cOGFF0bnzp0jSZKYO3dujB8/Pg455JD417/+FT/60Y9qujz4XrjjjjuiU6dO8cILL8S7774bW2yxRaWW33///WPp0qVRv379aqpw/Y499tg45JBDYtWqVfGf//wnbrrppnj00Ufjueeei5133rnG6oLvug3tP77r3nnnnahTp+bu8WvXrl1ceumlZaYXFhZWy/Zqen9hU/PAAw/EscceG82bN4+BAwdG586dY/bs2XHrrbfGfffdF3fffXc2k5k+fXqMHj06TjrppGjatGnNFs73ilAdKqFNmzbRuXPnmDZtWqnpM2bMiCRJ4sgjjywzL/NzJpBPq27dulG3bt0NKxjYZPXt2zd233337M8DBw6MoqKiuOuuu4Tq8C2YNWtWTJ8+PR544IH4+c9/HnfccUdccMEFlVpHnTp1Ij8/v5oqTGfXXXeNn/3sZ9mf99lnnzj00EPjpptuiptvvrkGK/v+WLlyZaxevbpGP1zh27Ux+o/qtHjx4mjUqFGN1rDmjUM1obCwsFTfWN1qen9hU/Lee+/FCSecEF26dImnn346Nttss+y8s88+O/bbb7844YQT4rXXXosuXbrUYKXfWLZsWdSvX98HZ99DXnGopH333TdeffXVWLp0aXbas88+G9ttt1307ds3nnvuuVi9enWpeTk5ObHPPvuUWs9DDz0U22+/feTl5cV2220Xjz32WKn5a4+p3qlTp3jzzTfjqaeeyn49sUePHtn2CxYsiCFDhkT79u0jLy8vtthii7jssstK1QJsupo2bRoNGjSIevUq/jz8gw8+iF/84hex1VZbRYMGDaJFixZx5JFHlvtshgULFsTQoUOjU6dOkZeXF+3atYsTTzwxPv/88wrXv3z58vjRj34UhYWFMX369I2xW1Br3XHHHdGsWbPo169fHHHEEXHHHXeUaXP33XfHbrvtFk2aNImCgoLYYYcd4tprr83OL29M9WeeeSaOPPLI6NChQ+Tl5UX79u1j6NChpf5fEfH/h0z5+OOPo3///tG4cePYbLPN4te//nWsWrWqyvt1wAEHRMQ3oV9ExD/+8Y/o169ftGnTJvLy8qJr165x0UUXlbuN559/Pg455JBo1qxZNGrUKHbcccdS+1sZL730UvTp0ydatmwZDRo0iM6dO8cpp5yyzmXS9nGZ/0M9++yzMWzYsNhss82iUaNGcdhhh8W8efNKtU2SJC6++OJo165dNGzYMHr27Fnma+UZaf6vlRl254orrohrrrkmunbtGnl5efHWW29V6TixaVpf/7HmeXLDDTdEly5domHDhtG7d+/48MMPI0mSuOiii6Jdu3bRoEGD+MlPfhLz588vs51HH3009ttvv2jUqFE0adIk+vXrV+b8zfQl7733XhxyyCHRpEmTOP744yu1P5kxidf3+0vENzcU7bHHHpGfnx9du3at8MO7tccYnz9/fvz617+OHXbYIRo3bhwFBQXRt2/f+Pe//11quUy/es8998Tvf//7aNeuXeTn58eBBx4Y7777bqX2a32++uqrGDJkSPb/Sq1atYqDDjooXnnllXUuN3HixGjYsGEce+yxsXLlynL3N9NPTZs2LX75y1/GZpttFk2bNo2f//znsWLFiliwYEGceOKJ0axZs2jWrFmce+65kSRJqe0sXrw4fvWrX2X7pK222iquuOKKMu1gU/PHP/4xlixZErfcckupQD0iomXLlnHzzTfH4sWL4/LLL49Ro0bFOeecExERnTt3zmYla//fIE3/9fHHH8cpp5wSRUVF2XZjx44t1SbTB919990xcuTIaNu2bTRs2DAWLVq0cQ8CmwR3qkMl7bvvvvHXv/41nn/++Wyo/eyzz8YPf/jD+OEPfxgLFy6MN954I3bcccfsvK23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vDzs3r0b3t7eSExMxMCBA2Xp16xZg88//xympqYYPHgwXnvtNeTl5SEzMxMJCQno2bMngIcB9VGjRiEuLg7t27fHhx9+CCMjI6SlpSEkJAQXLlzAypUr/7R6ExERERE1BoPqREREREQvqX/84x8AIAW9NS5cuAAHBwfZvuvXr6NLly6YMWOGLKj+008/4YsvvoC1tTWysrJgb28vHRNC4Pr167L3i4uLQ3BwMDZs2ABDQ0MAwIMHDzBs2DBERERg5MiRcHV1fdJVJSIiIiJ6YjinOhERERHRX1h9C5W2bt0aGRkZ6NixY715TZ48GZGRkcjNzYWdnR0AIDQ0FNHR0di0aROCg4PrfL2zszOuXLmCgoICmJqayo5lZ2fjnXfewbRp0zhanYiIiIieaxypTkRERET0Erhy5QrCw8Nl+9q0aYOjR4+iXbt2sv2//vorlixZgoMHDyIvLw/379+XHf/999+loPqJEycAAD4+PnW+/927d5GdnQ0bGxssW7ZM63hFRQUA4NKlSw2rGBERERHRn4xBdSIiIiKil4BarcaBAwcAAAUFBYiJicHMmTMxYMAAnDhxAmZmZgCAX375Bd26dUNJSQm8vLzg5+cHlUoFAwMDHDp0CIcPH5YF2YuLi6FQKGBtbV3n+9++fRtCCOTl5WkF92sqKyt7ArUlIiIiInp6GFQnIiIiInrJWFpaYvr06SguLsaiRYswd+5crF69GgCwatUq3L59G1u2bJGmi9GYMGECDh8+LNtnYWEhzZ1ua2tb63uqVCoAgKurK06dOvVkK0RERERE9CcyeNYFICIiIiKiZ2POnDmwsbFBVFQUcnNzATycJgaAbDFS4OGio1lZWVp5dOvWDQCQmppa53s1a9YMHTt2xMWLF1FUVPT4hSciIiIiekYYVCciIiIiekmZmppi5syZqKiowMKFCwFAmis9MzNTlnbp0qU4d+6cVh4TJkyAUqnE3LlzcfXqVdkxIQR+//13aXvy5Mm4e/cuxo0bp3Oal5ycHCm4T0RERET0vFIIIcSzLgQRERERET0dubm5cHBwkM2pXlN5eTkcHR2Rn5+PS5cuoaSkBG5ubnjllVcQEBCAli1b4tixYzh9+jT69OmDffv2ISMjA++9956Ux9q1azF58mQ0adIEgwYNgp2dHf744w8cOXIE/fv3l6aWEUIgODgYMTExsLa2hre3N2xsbHDjxg1cunQJx48fx7Zt2zBixIg/6dMhIiIiImo4jlQnIiIiInqJmZiYYPbs2aisrER4eDg6d+6M1NRUuLi4IDExEZs2bYKFhQWysrLQpUsXnXlMnDgRBw8ehJeXF5KTk7Fy5UqkpqbC2dkZAQEBUjqFQoHvvvsO27dvh5OTE/bu3Yu///3vSEtLg4mJCVauXAlvb+8/q+pERERERI3CkepERERERERERERERHriSHUiIiIiIiIiIiIiIj0xqE5EREREREREREREpCcG1YmIiIiIiIiIiIiI9MSgOhERERERERERERGRnhhUJyIiIiIiIiIiIiLSE4PqRERERERERERERER6YlCdiIiIiIiIiIiIiEhPDKoTEREREREREREREemJQXUiIiIiIiIiIiIiIj0xqE5EREREREREREREpCcG1YmIiIiIiIiIiIiI9MSgOhERERERERERERGRnv4fUkchj+rj2v8AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# First I chose only rows with sex \"Male\"\n",
        "# Next I caluculated mean of column age\n",
        "# And I rounded it to second place\n",
        "average_age_men = round(data[data.sex == 'Male'].age.mean(),2)\n",
        "\n",
        "print(\"Average age of men:\", average_age_men)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "Tsk8DqnNZBqs",
        "outputId": "2f7477f3-e334-4fa9-af05-d9a8c0240039"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Average age of men: 39.43\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotly \n",
        "\n",
        "fig = px.box(\n",
        "    data[data.sex == 'Male'],\n",
        "    x='age',\n",
        "    orientation='h', \n",
        "    height=300,\n",
        "    title=\"Age analysis of men\", \n",
        ")\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 317
        },
        "id": "F6tGgjjjZEBx",
        "outputId": "5e4fbc8b-f889-4166-c83e-5d6018ca2013"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script src=\"https://cdn.plot.ly/plotly-2.18.2.min.js\"></script>                <div id=\"4bca95c1-c7c6-42d4-889d-ff5f22f1efef\" class=\"plotly-graph-div\" style=\"height:300px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"4bca95c1-c7c6-42d4-889d-ff5f22f1efef\")) {                    Plotly.newPlot(                        \"4bca95c1-c7c6-42d4-889d-ff5f22f1efef\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"age=%{x}<extra></extra>\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\"},\"name\":\"\",\"notched\":false,\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"x\":[39,50,38,53,52,42,37,30,32,40,34,25,32,38,40,35,43,56,19,54,39,49,23,20,45,30,22,48,21,31,48,31,53,24,25,57,53,41,29,50,47,43,46,35,41,30,30,32,48,42,29,36,49,25,19,29,23,79,27,40,67,31,18,52,59,49,33,30,57,34,29,48,37,32,76,44,29,32,30,42,24,38,56,53,56,49,55,22,40,30,29,19,31,35,39,38,37,38,43,20,49,61,19,70,22,64,21,52,48,23,71,29,42,45,39,46,18,27,28,51,27,28,21,34,18,33,44,30,40,37,34,58,24,41,47,41,23,36,35,24,19,51,36,35,44,37,60,54,37,38,45,25,31,64,90,54,66,35,25,28,59,41,38,23,40,24,20,38,56,58,40,45,19,58,42,50,45,17,59,26,37,19,64,61,17,50,27,43,44,42,21,57,41,50,50,36,29,21,65,24,38,48,31,55,26,35,41,26,34,22,24,77,39,43,35,76,63,58,66,41,26,47,55,53,17,30,49,19,45,26,38,36,33,43,67,56,31,33,26,33,46,59,38,65,40,26,62,43,43,22,28,56,17,40,45,44,23,46,38,54,25,29,27,46,34,44,25,52,20,50,28,41,28,46,32,46,31,35,52,34,20,17,29,25,36,23,63,47,80,17,27,33,34,23,29,24,44,27,20,51,20,45,60,42,40,38,23,32,44,54,32,50,52,38,49,60,22,35,30,17,22,27,33,43,28,52,63,59,38,46,33,20,23,72,23,62,52,24,19,43,49,32,34,28,20,34,38,50,24,37,44,42,38,44,26,36,67,39,29,44,29,27,27,35,21,28,46,35,30,28,54,47,52,20,43,29,47,24,51,17,37,29,18,26,65,57,59,27,31,29,18,52,57,42,60,31,27,42,25,32,19,42,35,51,29,36,24,24,26,39,30,50,52,46,29,47,30,34,38,33,49,43,58,21,33,47,52,26,60,21,36,31,50,71,58,30,20,24,35,34,44,43,44,35,27,43,27,42,24,48,17,33,50,17,32,31,58,29,37,34,23,66,41,26,54,42,20,25,35,31,36,21,31,40,45,60,18,28,36,40,33,34,33,41,29,50,42,43,31,65,26,35,43,42,55,26,17,55,29,46,29,22,58,23,39,27,54,37,34,53,51,31,32,37,31,55,23,34,43,54,43,40,41,61,18,59,48,41,18,60,22,61,25,46,43,43,24,68,32,50,33,64,20,22,43,22,17,56,64,47,48,29,32,57,43,24,42,26,73,55,25,24,63,17,35,51,55,43,37,39,31,46,48,34,59,43,48,28,33,21,17,39,29,44,24,71,51,55,41,39,32,27,38,63,23,33,54,35,37,42,40,47,30,28,63,51,27,35,37,24,37,53,27,38,53,34,23,39,67,81,21,23,25,64,32,37,51,42,51,37,37,41,33,31,35,38,20,43,44,43,51,41,44,33,33,42,25,32,28,35,55,48,34,67,37,44,40,34,60,45,44,46,41,50,47,35,56,45,48,40,39,20,50,31,58,66,39,54,26,34,50,42,38,48,22,19,41,48,34,56,30,45,48,20,27,76,19,66,37,44,28,23,36,61,53,30,52,38,32,30,49,45,43,61,54,49,38,35,36,30,38,37,17,19,22,49,58,41,31,30,32,29,42,38,41,44,31,34,21,22,62,68,45,39,41,31,29,41,41,35,37,60,36,41,53,29,34,36,35,63,41,28,42,31,46,35,27,41,20,40,56,33,46,61,50,25,24,51,43,34,46,26,20,44,51,33,54,65,88,40,25,20,47,58,50,24,50,36,32,44,51,65,27,51,48,39,28,51,59,50,45,30,37,44,33,61,61,38,22,36,44,37,54,20,48,31,38,45,48,53,38,36,42,75,46,43,35,65,38,54,49,51,48,30,28,29,57,43,37,30,53,22,38,50,37,44,27,51,46,67,47,42,52,65,21,46,47,57,37,50,58,39,46,36,26,21,41,44,22,29,36,30,37,29,43,45,32,35,35,41,48,20,27,55,39,17,47,25,23,57,51,39,40,38,28,47,59,26,61,36,62,34,32,39,58,61,40,31,36,23,33,52,52,20,26,61,46,44,54,65,25,23,25,19,37,45,37,40,19,17,40,43,42,54,54,51,50,55,31,24,30,37,33,43,39,53,52,28,32,31,19,40,41,53,46,45,19,57,32,35,28,47,50,61,36,38,56,50,45,44,51,44,25,60,17,49,46,25,22,41,48,46,30,39,40,70,35,24,40,61,25,69,47,33,43,58,53,26,43,31,30,37,32,45,53,27,28,32,36,19,37,49,20,64,60,32,58,42,42,25,30,37,66,59,44,46,25,17,47,26,20,33,48,52,47,40,49,34,43,35,62,39,59,69,27,17,38,41,17,62,44,23,41,25,18,45,51,29,33,43,20,44,24,19,49,52,38,24,32,26,31,29,39,48,40,51,28,65,26,20,55,38,33,19,24,30,57,46,67,47,39,50,45,54,26,60,46,32,38,48,44,44,61,51,39,40,27,37,57,34,54,42,48,65,33,30,35,19,36,26,34,41,44,47,18,55,49,44,43,62,44,50,35,52,46,44,43,51,20,43,64,24,27,32,26,43,20,37,37,44,19,44,51,35,44,54,24,35,56,24,43,47,49,38,31,41,35,38,51,38,40,44,55,29,31,24,46,26,35,27,64,26,45,57,30,43,32,38,53,47,40,51,65,28,24,32,71,26,29,50,51,31,23,59,17,43,38,37,29,39,31,39,30,43,31,33,32,28,50,28,38,41,50,59,32,73,52,57,35,51,49,22,46,52,43,25,31,19,19,51,51,21,46,56,58,70,30,45,23,34,24,53,45,32,21,63,23,28,29,26,23,46,61,25,38,23,46,32,40,31,31,32,21,63,44,42,23,24,42,27,27,55,22,68,26,34,68,42,45,26,25,23,29,31,39,44,26,36,38,39,52,39,41,42,47,28,25,38,44,30,30,45,20,57,43,25,90,38,50,39,56,37,29,40,44,77,30,21,66,40,68,32,32,51,50,49,28,21,55,45,35,31,26,30,33,40,23,48,39,49,36,60,29,54,34,26,47,50,44,49,37,51,35,66,39,27,24,28,30,34,57,39,23,36,32,36,47,35,44,32,34,47,27,50,49,36,23,41,27,53,17,23,53,28,53,18,24,30,37,34,19,20,42,46,34,45,35,63,51,36,25,25,19,47,34,42,37,51,25,37,42,29,35,51,22,49,26,52,61,52,43,35,17,42,58,66,30,25,37,39,23,25,24,35,47,47,46,49,34,25,58,31,29,28,30,21,44,66,26,33,57,47,41,17,61,47,45,32,38,41,51,42,60,37,54,34,41,35,63,46,47,24,29,50,30,32,39,49,44,65,28,45,34,49,42,60,64,33,45,22,23,31,23,51,59,45,35,33,37,36,38,54,51,45,51,58,20,44,44,63,26,42,29,38,25,27,31,24,43,23,44,40,38,23,41,48,23,42,18,31,29,41,46,42,46,18,52,21,22,41,43,26,90,45,42,20,53,56,42,58,41,58,62,65,40,45,22,23,63,39,37,25,56,32,32,23,35,51,41,22,36,45,23,44,29,36,18,62,25,37,33,28,22,47,67,20,34,26,36,28,22,33,25,37,36,61,74,57,55,54,42,17,39,56,33,37,27,36,30,22,35,20,25,34,20,70,57,54,34,48,45,27,51,33,50,44,39,55,35,32,66,29,24,61,34,55,20,36,60,61,35,42,26,31,39,32,32,61,23,19,31,35,20,60,39,31,38,26,64,46,69,39,42,50,39,29,21,32,69,55,41,44,18,74,40,27,75,52,36,76,18,38,50,32,57,20,31,61,57,19,59,34,31,42,47,47,29,27,51,35,36,46,65,43,47,49,18,27,34,24,34,47,34,25,54,24,41,51,36,25,55,58,46,19,24,53,24,29,56,54,34,34,32,67,47,52,28,41,57,36,33,46,33,77,39,67,41,60,32,56,18,31,21,45,45,25,31,19,41,27,31,40,41,25,42,42,45,52,34,32,54,42,32,55,66,52,52,27,47,46,50,43,34,19,37,49,26,48,39,70,20,38,51,43,38,18,23,34,62,33,26,55,61,37,78,49,55,42,61,64,46,24,27,47,18,48,50,56,56,18,27,52,50,19,20,24,42,51,39,34,19,30,21,62,25,32,26,37,38,47,42,30,43,34,46,32,47,55,53,43,80,43,22,40,48,56,34,51,24,41,22,20,31,55,49,26,27,40,59,61,43,61,30,44,34,28,47,26,35,36,47,35,33,18,47,53,24,27,50,36,41,32,22,22,18,39,35,23,54,36,63,39,57,58,40,38,44,33,41,37,50,48,36,37,27,46,17,40,47,25,35,30,30,25,28,43,47,46,29,25,56,31,22,31,53,71,39,36,56,48,46,29,47,67,43,42,31,55,28,30,33,40,63,23,56,42,40,35,35,43,29,23,50,34,19,37,75,24,28,31,18,45,31,27,34,53,60,48,23,68,42,37,38,27,49,27,36,36,35,36,53,34,61,29,37,22,43,55,64,46,29,44,20,62,35,39,39,38,72,31,55,19,49,26,35,37,42,28,35,40,55,21,51,57,51,38,45,28,42,22,28,47,40,30,61,44,46,27,40,25,61,54,31,53,39,57,30,35,42,34,59,69,40,21,43,36,41,72,20,34,35,28,27,20,32,38,23,28,40,74,28,23,20,20,51,48,60,48,48,39,65,28,46,42,61,28,24,45,24,44,20,35,52,25,19,35,21,55,44,25,26,25,43,45,48,60,45,51,22,44,27,47,32,40,41,47,55,38,24,64,56,41,33,33,21,59,58,43,21,45,41,53,38,31,41,43,41,41,36,20,30,21,29,35,35,43,39,28,30,20,51,62,37,28,23,50,31,45,23,47,32,82,49,74,22,32,34,59,36,47,53,25,44,42,58,39,32,45,52,56,36,41,30,44,41,29,25,42,48,26,46,38,49,30,49,47,44,31,19,34,26,38,36,31,30,37,30,38,30,33,52,35,59,27,31,45,27,19,29,34,32,49,74,62,39,24,76,61,47,50,27,43,39,56,26,47,47,45,26,20,37,37,43,63,45,36,26,41,26,46,24,41,20,36,30,25,49,26,25,41,35,36,51,39,36,40,61,38,20,46,21,35,39,38,28,32,46,40,44,38,51,46,51,45,44,33,61,47,34,53,39,58,66,26,41,47,31,52,29,42,41,34,52,61,39,46,39,22,43,37,23,39,54,46,36,37,50,38,63,53,23,20,46,28,59,41,33,56,50,41,39,35,38,31,17,31,25,39,50,50,63,31,18,34,18,46,59,41,42,55,69,30,21,18,35,20,26,46,40,32,23,25,37,37,47,51,30,59,45,52,21,31,29,24,17,46,34,63,21,18,49,36,47,36,39,57,39,24,56,41,51,47,56,51,57,24,57,38,35,58,19,68,43,53,37,21,63,31,29,43,25,23,23,40,32,72,61,42,73,41,65,52,23,26,37,17,31,63,31,35,36,32,60,41,26,45,22,39,63,61,31,25,21,18,34,44,31,44,42,43,33,30,55,37,32,62,31,51,45,47,31,75,43,39,30,27,44,40,33,20,38,46,51,24,30,70,26,34,46,37,28,44,58,53,23,26,21,43,22,41,36,27,42,41,21,44,29,21,46,38,43,54,38,18,30,37,32,30,18,27,49,46,37,30,22,25,35,56,41,47,46,28,39,68,37,24,19,37,28,52,49,80,63,63,66,33,49,20,33,26,52,33,47,23,53,19,26,53,65,68,30,28,54,30,22,25,19,32,51,25,26,29,51,40,24,34,19,21,21,23,37,40,57,48,24,43,52,49,45,62,29,20,37,43,28,45,22,20,60,22,43,51,41,62,50,54,68,23,24,30,38,22,30,61,36,51,54,51,63,51,35,46,53,26,43,39,50,54,19,29,20,53,18,22,23,69,56,53,28,28,33,61,49,48,32,51,27,19,47,25,72,29,57,50,45,52,53,36,46,75,32,28,38,36,28,28,43,38,39,40,24,30,44,40,42,39,43,62,30,39,25,48,59,44,36,40,67,24,50,48,55,29,64,49,20,37,52,22,19,52,37,37,28,63,19,23,31,35,40,35,35,28,28,37,72,25,31,44,20,57,44,29,36,40,21,21,34,35,35,29,23,51,28,44,29,31,27,34,50,27,46,40,28,58,66,25,90,35,59,19,39,19,52,50,36,34,23,33,27,31,21,31,59,40,19,47,34,57,51,21,30,46,52,23,43,34,65,50,29,25,49,44,25,65,56,39,24,41,40,55,44,45,29,43,30,31,38,66,63,21,33,20,47,33,66,28,25,37,31,49,22,31,54,46,49,54,41,49,23,67,23,22,66,25,35,67,30,34,21,18,21,76,51,18,34,55,33,36,21,30,41,34,24,41,33,27,51,26,38,54,26,75,29,61,22,34,42,24,42,27,20,68,28,35,41,44,46,27,32,46,26,49,32,32,49,36,22,42,22,44,22,59,30,64,59,43,23,49,32,33,47,41,34,25,46,37,41,44,30,54,35,35,35,19,56,40,45,26,32,35,22,25,64,46,30,63,51,68,36,28,41,47,33,34,46,36,33,51,56,53,22,33,28,53,37,50,22,21,25,43,32,42,33,29,29,35,31,33,67,46,17,35,48,27,64,53,29,32,30,34,19,39,28,24,31,47,45,52,52,60,43,38,40,35,52,53,35,29,29,58,64,40,52,38,39,68,24,40,43,45,18,57,52,56,26,21,48,33,31,26,37,20,28,73,49,40,18,41,46,64,50,53,21,22,39,30,25,43,55,20,28,29,60,27,50,41,46,21,40,49,39,35,52,25,24,29,21,43,47,61,21,35,60,57,56,19,43,22,35,48,31,57,27,34,22,40,37,40,53,38,25,51,22,18,41,67,47,24,32,25,55,31,32,18,34,19,36,52,22,57,54,27,26,25,47,23,42,39,55,46,53,48,30,41,38,77,42,20,32,27,25,40,18,49,47,24,58,29,30,19,59,41,28,46,40,30,44,58,21,35,45,30,33,36,30,33,53,38,34,45,33,22,45,17,50,38,33,46,64,62,32,17,61,39,35,29,27,51,31,29,56,26,46,32,28,27,23,59,26,36,33,48,27,19,38,26,45,32,22,34,60,24,31,37,23,31,29,43,23,29,22,25,29,25,33,30,55,36,28,42,51,65,23,60,47,70,32,37,31,32,31,65,30,63,40,34,44,36,31,22,28,23,49,39,67,19,30,24,46,37,41,40,32,58,48,34,46,36,59,29,50,25,48,36,42,49,33,40,59,30,24,51,34,47,47,34,52,53,26,57,31,20,37,25,36,45,34,47,60,75,43,18,57,43,18,48,35,41,32,33,58,22,42,39,35,49,25,42,37,25,29,26,45,43,28,51,46,38,20,36,37,44,33,45,47,54,17,44,41,35,18,47,41,26,30,42,28,48,44,34,32,25,47,18,38,33,41,46,38,24,49,44,63,40,45,49,65,41,60,31,45,51,23,34,19,51,46,30,37,56,29,46,55,45,34,45,41,22,34,27,29,47,43,33,29,23,23,39,31,47,26,47,36,40,19,63,42,76,38,33,48,56,32,33,22,44,42,45,57,59,52,29,63,20,42,50,28,55,44,23,55,48,46,26,30,32,52,52,57,21,22,26,26,44,40,22,62,44,33,45,18,32,20,68,61,48,45,39,53,37,40,52,33,28,25,53,44,38,34,29,39,46,37,47,62,54,90,33,57,25,44,52,21,31,39,29,49,34,22,32,46,50,34,20,60,40,52,42,25,41,40,38,58,28,28,17,39,28,42,28,47,27,32,27,29,22,41,49,58,41,26,21,37,23,42,25,18,25,43,21,65,33,27,32,38,40,63,76,43,76,41,17,30,30,22,47,34,25,28,24,44,18,54,51,43,51,24,20,52,57,49,38,39,35,58,48,18,29,52,63,30,23,28,20,18,27,51,23,57,59,38,20,35,50,45,44,23,34,39,19,54,40,44,25,32,36,47,35,77,30,35,49,53,41,57,42,29,26,41,36,28,42,32,32,37,39,49,28,31,29,59,52,27,45,58,53,23,35,49,25,74,19,28,61,30,20,35,23,19,30,26,34,35,36,33,19,18,60,28,43,49,25,17,41,30,90,25,27,29,55,43,46,59,26,61,23,35,42,43,17,23,35,50,25,32,29,57,90,31,22,37,74,67,59,28,45,63,32,34,39,29,22,46,39,45,37,33,33,48,32,18,35,44,55,34,33,36,28,39,49,22,31,44,19,37,60,23,44,23,19,51,41,61,22,49,35,38,29,56,62,27,25,33,43,53,56,25,60,21,29,25,32,27,43,35,39,44,27,69,20,62,31,25,54,44,34,21,50,33,44,33,21,43,57,51,29,51,35,18,58,57,38,27,38,30,46,60,30,47,42,44,33,54,58,45,59,44,44,33,27,33,32,31,57,30,30,29,60,31,26,24,29,44,34,59,23,23,39,39,57,39,53,23,35,23,32,49,36,41,54,31,46,56,21,51,47,42,25,37,35,40,57,38,46,30,45,32,35,56,50,25,41,33,37,56,19,53,33,72,54,51,30,30,32,30,35,32,76,60,60,55,24,53,71,45,59,44,31,51,51,23,49,56,39,39,43,31,44,43,37,41,43,47,38,46,19,24,39,40,21,20,45,46,26,47,34,49,20,28,30,23,20,31,46,60,30,27,56,41,61,60,33,48,56,50,57,37,42,54,50,48,46,52,34,26,33,43,35,35,45,43,51,35,47,22,49,20,27,27,23,47,17,76,25,44,53,34,45,34,35,36,37,53,38,65,49,24,54,19,46,58,37,41,30,48,50,18,21,47,39,67,19,18,36,23,40,73,35,31,20,27,66,66,48,51,25,50,51,34,30,59,41,36,37,28,19,28,62,36,58,42,43,29,32,64,19,30,19,33,23,29,30,34,60,48,25,55,22,25,73,39,33,33,42,43,57,42,22,42,39,20,47,71,52,36,53,26,34,35,34,56,23,27,51,31,59,38,34,42,45,21,64,32,45,53,42,49,62,28,29,32,37,38,44,19,22,58,43,32,28,38,20,24,40,23,79,30,52,41,33,50,51,41,34,28,31,22,65,39,23,29,57,51,34,52,29,29,25,55,41,27,78,40,26,43,25,43,25,52,42,44,65,56,33,45,37,56,31,26,26,42,45,47,22,33,37,48,57,58,52,29,23,46,30,32,56,33,19,34,55,31,31,51,59,40,54,25,18,41,37,41,54,39,41,57,33,56,60,22,58,45,48,38,23,63,51,44,46,64,32,31,36,30,54,59,44,53,58,39,27,37,33,20,55,39,25,31,51,24,23,45,47,79,30,27,40,60,58,71,50,29,60,33,33,32,21,42,37,28,37,49,40,40,49,35,33,24,43,36,84,23,41,55,30,46,55,34,54,27,25,52,31,66,90,67,22,32,55,21,44,23,56,24,43,26,42,53,51,40,27,24,23,42,38,63,31,21,55,35,51,38,52,70,53,59,39,19,32,34,50,18,37,43,77,30,33,51,58,22,33,27,35,37,17,50,31,23,29,38,30,20,43,25,39,48,24,20,57,66,36,46,34,31,70,32,46,42,21,23,50,44,29,36,49,29,75,51,61,50,59,22,56,35,57,47,50,30,46,48,61,33,40,58,46,35,64,41,18,44,33,20,38,20,22,20,20,33,54,25,18,49,17,34,37,49,37,20,34,30,44,42,39,19,41,29,34,38,27,17,33,24,28,21,60,37,26,40,55,33,62,26,47,52,46,67,34,39,54,24,38,36,43,39,26,37,43,54,53,57,28,29,21,51,46,50,31,36,38,37,39,18,33,23,25,51,35,54,66,37,29,45,61,49,51,34,20,49,26,63,36,63,40,49,35,25,34,28,44,33,21,55,75,26,48,28,29,37,40,34,55,31,32,26,26,44,49,40,58,61,28,26,42,41,45,39,41,33,23,35,41,33,22,77,19,28,19,43,39,33,40,41,23,48,38,37,45,29,23,34,33,27,33,26,56,46,90,72,29,49,57,33,24,49,54,31,43,19,46,43,58,43,37,21,52,34,34,29,39,35,18,47,46,43,51,18,39,58,19,36,40,25,33,63,51,34,33,25,30,19,36,60,28,42,54,36,55,57,56,24,35,51,34,46,22,37,48,61,40,52,18,27,48,63,38,31,40,41,31,32,30,39,28,21,44,22,42,36,39,61,31,39,34,21,75,21,81,54,34,54,41,63,24,47,31,41,53,44,37,28,71,22,33,20,20,34,21,22,18,22,42,39,31,47,47,64,38,59,35,33,44,22,56,44,32,17,44,32,35,27,32,68,24,48,28,60,23,23,52,35,41,37,63,24,43,36,32,36,17,41,41,51,54,19,47,46,35,52,73,17,43,19,41,49,52,57,33,56,42,17,39,37,22,27,59,44,53,26,48,39,54,36,60,50,48,30,65,31,46,62,41,38,45,45,46,24,35,29,56,60,25,33,21,21,46,44,31,26,37,33,38,47,46,47,47,69,32,51,36,29,32,27,31,46,39,69,54,34,24,63,29,64,44,26,55,37,33,31,58,25,18,26,29,45,32,26,56,49,37,23,28,27,49,41,28,38,47,28,44,41,50,73,21,64,52,24,44,27,61,58,23,21,35,30,47,55,22,30,24,49,23,38,49,46,56,24,58,28,22,46,30,56,32,56,50,24,31,44,55,19,50,53,41,37,46,45,30,57,32,43,34,33,23,39,19,47,46,47,38,58,32,20,25,25,36,69,21,37,21,25,34,33,47,48,42,34,20,41,45,22,23,24,20,32,52,36,18,26,24,42,39,68,54,24,64,36,44,38,56,24,42,67,32,44,65,37,27,64,28,63,40,33,40,25,45,30,63,27,31,33,39,42,19,31,53,47,32,23,27,18,30,40,26,27,22,37,54,36,44,38,18,26,48,35,40,42,30,36,50,38,35,45,41,28,49,47,30,37,27,45,37,54,47,35,33,28,25,33,53,21,31,63,21,59,19,31,35,32,38,60,21,23,67,38,20,26,58,31,37,53,33,35,60,38,44,27,34,20,49,37,52,46,43,48,36,37,18,42,60,37,44,32,42,17,28,28,19,49,25,42,47,26,35,37,39,42,31,36,36,50,37,25,29,50,42,32,48,20,50,18,60,60,42,30,59,24,39,44,21,23,22,55,23,46,47,42,18,37,34,51,34,28,56,46,27,18,36,39,36,24,34,33,53,34,23,39,54,32,27,54,38,21,40,21,38,41,70,45,83,18,31,33,27,41,53,66,31,30,22,47,74,37,35,51,51,32,29,42,22,62,21,46,25,29,29,33,50,40,28,54,29,71,27,64,36,39,26,31,43,27,24,50,30,24,29,31,37,28,36,34,44,34,28,38,64,35,36,34,27,22,36,51,63,51,42,25,49,32,28,19,18,39,36,45,30,23,43,25,50,61,35,56,55,40,35,22,20,46,43,39,30,25,44,41,43,50,38,20,37,31,31,44,31,65,17,60,31,53,44,43,40,38,57,32,64,40,23,27,23,58,47,33,27,46,24,40,31,37,30,31,32,27,35,27,30,33,34,29,39,37,40,53,58,29,33,57,69,49,34,70,37,58,41,47,58,50,84,47,19,24,48,53,58,22,36,37,43,21,49,35,28,51,19,56,20,33,54,22,23,69,41,17,31,50,23,18,53,42,23,47,53,28,28,44,58,36,39,25,39,52,26,32,49,40,52,33,19,37,33,48,43,32,28,47,37,26,55,31,39,54,38,53,42,47,59,25,58,61,33,51,55,29,31,37,43,23,41,45,25,33,38,68,42,35,33,45,57,36,53,28,38,37,30,29,31,36,59,33,29,49,50,33,28,32,47,44,24,51,52,28,47,43,25,43,59,38,37,26,73,30,33,46,18,50,62,39,22,29,38,49,33,28,37,30,49,29,36,21,30,40,43,45,43,32,21,23,52,19,54,29,17,30,23,45,20,26,49,26,47,31,23,35,46,28,58,35,33,34,67,37,48,34,24,50,40,33,51,30,41,47,27,40,36,37,38,40,20,33,66,32,23,44,20,44,41,45,40,34,23,48,42,22,60,42,34,39,42,53,44,46,22,24,47,64,62,52,42,38,55,32,36,27,47,33,48,23,31,26,61,50,49,57,19,18,54,35,27,50,43,28,47,21,19,40,55,67,31,30,59,29,17,52,32,44,32,45,30,41,48,31,34,25,41,35,34,44,32,25,23,36,53,27,30,18,23,19,20,52,39,19,41,24,22,45,45,48,42,35,59,31,27,29,40,31,23,60,27,31,29,22,63,28,79,39,29,60,44,47,59,37,21,43,34,22,38,24,39,26,24,22,36,18,38,43,56,42,35,73,37,66,51,31,28,48,29,36,35,31,48,36,40,57,65,42,47,62,27,76,29,32,22,23,44,41,30,23,32,50,35,23,25,30,52,49,46,39,45,43,62,48,23,26,36,38,18,56,39,46,36,42,50,17,50,35,27,27,23,51,47,40,36,20,25,64,53,50,51,20,47,25,50,43,21,58,30,72,33,46,33,31,28,51,25,61,26,35,33,35,47,64,34,62,53,31,26,56,33,45,46,26,43,40,61,85,34,24,45,21,50,41,45,34,28,19,26,48,69,28,31,51,43,31,31,45,25,31,56,26,40,37,38,19,82,40,28,23,19,67,43,20,40,47,40,78,49,66,59,35,36,23,39,40,29,38,63,58,44,21,41,38,75,29,52,28,30,30,40,26,32,26,25,27,24,38,35,29,36,47,21,38,66,45,18,37,33,27,24,44,48,29,29,29,32,79,34,46,50,29,51,34,29,49,77,35,42,35,62,33,38,26,33,35,42,43,69,39,37,28,26,32,43,49,28,35,35,45,23,45,35,45,28,64,59,41,21,51,22,53,23,20,40,28,32,20,62,31,31,24,43,21,30,18,24,49,19,48,32,23,50,30,32,39,57,22,34,21,37,60,23,32,57,65,25,30,19,23,44,43,34,32,49,42,30,33,36,17,20,32,33,60,46,29,36,27,50,33,41,22,25,28,25,28,39,59,38,34,31,48,65,45,80,33,38,44,40,70,32,22,52,67,40,62,64,29,28,25,47,26,18,52,36,41,56,53,22,31,36,46,46,43,30,41,50,37,22,51,40,46,27,31,30,50,29,28,47,26,53,23,42,36,52,49,36,20,55,47,33,36,55,21,48,30,24,20,44,57,57,33,61,39,28,31,36,50,30,20,90,47,34,47,34,34,35,39,32,28,41,32,47,41,35,24,46,69,31,23,37,18,53,54,24,36,38,33,66,45,37,31,46,40,34,56,46,30,33,57,34,40,33,22,45,18,30,34,36,36,48,41,47,41,61,64,35,44,32,63,42,24,35,40,46,29,42,36,26,57,55,35,47,34,44,27,21,29,25,73,45,43,47,22,76,24,54,30,33,30,40,27,30,19,46,51,37,41,65,57,44,32,27,59,43,38,61,24,25,44,90,44,36,63,60,49,47,42,53,20,43,32,43,31,24,39,29,20,23,53,20,58,51,25,41,58,25,50,32,31,37,62,40,56,41,56,20,34,45,30,51,27,38,39,57,47,77,43,46,41,30,30,22,25,50,37,23,27,44,25,18,22,27,41,24,34,46,32,41,29,51,58,52,37,54,36,23,39,20,28,67,56,32,42,33,37,33,59,21,58,24,26,26,25,21,24,38,41,37,54,27,19,25,47,40,38,25,23,29,32,52,33,19,68,41,29,40,51,49,38,57,35,66,27,77,70,59,25,44,37,47,32,53,19,23,21,36,72,51,32,37,21,26,34,36,31,32,54,46,33,32,42,30,37,28,25,44,40,36,21,39,24,46,47,47,35,60,37,48,22,23,58,24,38,49,47,50,47,47,36,28,46,32,33,60,53,39,23,50,38,41,72,39,35,26,52,41,58,34,36,19,60,35,26,17,59,53,35,34,36,36,21,35,59,22,50,42,52,46,23,54,33,48,19,36,35,33,48,19,64,51,20,28,52,29,51,36,43,52,50,25,47,32,37,40,34,35,26,49,63,46,31,22,52,34,31,22,38,35,27,27,19,69,20,34,49,27,45,64,41,49,51,69,43,32,27,38,62,46,17,36,23,58,40,66,42,53,25,20,32,28,18,43,29,32,31,60,42,46,38,54,46,27,50,38,25,19,60,45,21,42,38,22,19,23,30,62,26,42,33,54,33,24,54,58,39,32,27,27,20,64,45,26,36,58,39,35,38,50,28,55,29,33,32,45,23,31,59,32,22,26,23,47,44,27,20,40,44,62,47,44,26,36,52,48,33,25,62,38,46,46,52,63,49,61,33,46,40,43,24,29,19,24,40,31,58,40,31,49,24,46,58,65,48,41,52,26,52,19,27,48,60,28,69,40,41,38,21,38,42,42,32,24,26,49,65,34,39,19,21,35,48,33,43,45,41,27,50,29,41,55,36,23,22,58,37,55,55,28,53,32,60,34,49,29,43,31,38,42,62,24,36,57,46,33,42,62,64,43,29,43,39,69,44,25,47,17,41,19,57,25,37,29,32,35,52,31,40,21,49,70,22,29,37,29,38,27,58,20,51,37,22,56,27,52,28,44,30,55,51,43,36,36,30,24,62,41,49,36,52,52,26,51,18,30,46,48,49,35,78,61,36,17,27,64,33,32,42,23,51,31,44,44,38,29,47,30,37,27,44,31,34,39,57,24,21,34,61,67,49,58,37,39,45,30,49,36,23,55,35,45,36,32,39,47,49,52,44,36,59,30,50,28,25,45,20,40,52,34,37,17,35,33,64,50,18,67,46,60,56,21,46,25,40,43,29,18,28,53,63,61,35,43,48,51,34,44,61,37,36,34,63,47,31,27,28,31,43,50,68,37,26,22,59,30,30,50,38,29,76,21,43,32,37,64,49,44,68,34,35,24,25,36,29,37,32,47,44,19,22,51,69,73,33,28,27,22,67,31,43,27,36,63,25,41,26,30,32,48,39,40,40,19,47,41,40,25,26,17,73,19,37,26,47,47,29,33,20,26,41,67,43,35,55,34,38,21,37,54,49,51,29,68,56,30,41,35,21,49,45,27,28,35,33,35,19,28,61,39,42,22,18,49,33,28,42,20,50,32,29,46,32,43,39,18,42,32,52,36,38,23,71,56,28,45,28,62,39,24,58,48,57,31,25,29,34,30,29,22,42,19,42,20,51,50,27,26,61,20,51,52,33,31,56,19,29,43,43,21,41,45,47,80,19,45,17,37,17,18,57,36,51,38,77,51,24,35,35,51,28,37,43,23,27,59,28,24,20,54,65,18,38,29,36,55,22,24,36,68,49,18,60,43,39,42,48,56,28,39,40,38,59,51,19,42,43,45,34,21,34,55,29,45,42,90,44,34,39,29,49,52,54,31,27,24,30,34,28,51,33,29,33,37,20,31,41,42,54,21,52,19,33,17,31,74,35,45,42,44,52,24,56,27,47,40,38,22,41,55,25,32,33,17,27,55,63,51,67,41,54,59,19,31,42,51,23,27,25,30,33,28,45,25,37,59,64,39,24,46,35,31,35,35,40,48,70,37,24,35,43,18,54,36,27,23,61,39,33,22,31,26,44,73,35,47,29,32,41,45,33,32,42,38,38,37,24,27,29,46,44,46,38,38,22,33,50,31,42,27,34,64,43,39,47,56,25,52,43,22,28,28,31,39,39,28,67,50,34,54,44,37,38,59,59,51,60,29,42,28,50,22,63,24,19,23,20,63,28,51,17,18,48,65,27,39,51,49,58,36,63,63,44,48,51,23,31,38,19,56,31,46,39,56,42,53,18,27,31,52,46,30,48,37,29,49,24,74,56,39,21,43,47,27,45,30,33,37,21,42,48,54,36,56,37,41,55,39,67,38,47,49,20,51,19,33,41,90,78,30,35,46,38,24,49,19,46,41,28,21,24,36,23,38,29,38,47,40,21,23,44,37,51,57,36,48,24,43,27,19,32,35,26,45,20,33,35,31,20,44,33,50,28,60,38,38,57,50,69,51,42,53,37,37,38,52,48,30,37,47,20,41,32,54,45,62,33,42,43,50,42,36,22,30,33,43,38,34,40,41,57,65,52,59,18,48,45,24,27,40,44,24,71,17,57,71,37,33,41,42,34,24,23,33,58,25,53,25,31,18,40,68,33,22,50,24,37,34,77,46,26,57,28,36,29,36,19,48,49,55,42,30,32,37,35,30,55,36,33,56,26,37,28,38,76,35,39,64,45,27,25,39,37,26,33,34,41,38,36,38,46,54,24,23,58,26,45,30,34,67,29,40,61,45,58,52,34,25,46,43,25,65,37,29,44,37,35,32,44,25,47,25,35,68,50,35,24,21,17,27,50,51,27,23,42,32,45,22,48,22,34,38,49,23,24,44,34,29,35,48,38,59,31,34,51,25,30,25,48,36,27,30,30,30,35,39,74,32,25,47,47,33,22,56,30,53,43,23,43,30,22,38,54,23,46,59,28,18,41,42,58,24,39,25,53,63,21,45,72,51,21,46,31,43,21,44,56,20,36,52,56,51,29,36,29,58,37,30,29,48,50,35,54,20,30,26,38,33,60,30,30,23,27,20,39,33,65,34,23,25,31,28,52,31,37,45,45,44,32,35,53,54,43,30,49,52,60,45,48,28,61,48,32,27,38,33,36,38,22,30,55,46,39,50,47,46,23,45,55,68,39,37,19,20,33,21,72,58,43,21,26,27,34,34,58,53,52,36,48,35,49,43,33,49,17,25,69,41,31,57,49,40,55,47,50,36,17,30,79,42,67,36,35,21,46,45,20,17,40,28,33,55,52,36,37,43,24,47,39,47,27,29,37,38,28,35,18,36,52,29,20,42,40,32,29,26,25,37,26,23,20,36,24,38,44,46,37,44,31,32,20,47,40,46,59,47,47,45,45,73,48,38,37,62,26,46,65,29,37,60,33,66,32,29,70,74,37,51,50,42,26,27,58,30,32,66,23,35,50,30,84,35,45,46,50,37,53,27,22,29,35,29,26,31,30,20,37,54,78,57,37,43,38,70,40,40,20,29,24,58,40,39,35,45,62,22,23,39,51,73,24,24,52,43,67,31,36,43,31,23,26,48,40,49,63,66,60,19,26,75,65,50,26,45,19,29,21,59,50,41,40,19,30,42,37,28,50,20,21,21,45,49,53,41,41,53,40,28,32,44,28,29,44,37,62,31,20,56,36,29,34,39,29,38,23,37,42,45,46,41,49,37,31,22,47,17,39,23,41,36,52,41,38,24,39,36,29,17,47,61,40,29,26,46,56,23,32,34,49,54,39,41,47,27,23,44,25,52,19,22,53,45,69,49,44,56,37,37,34,36,54,28,28,73,37,38,50,69,43,42,39,30,30,37,50,22,21,33,37,34,29,29,28,73,39,27,47,47,42,69,18,44,41,47,43,33,39,27,27,79,24,42,63,47,42,63,65,30,62,46,20,28,61,30,29,57,41,23,30,43,64,56,32,55,17,37,50,41,71,51,53,40,54,33,36,42,52,30,22,47,26,36,35,46,30,38,38,36,47,59,52,37,51,67,58,49,35,18,47,31,22,32,39,51,48,39,35,40,19,28,18,23,47,42,25,30,29,46,69,50,19,59,45,47,29,24,28,35,48,49,42,36,41,35,41,48,47,51,22,49,26,35,34,51,26,32,38,44,26,38,62,20,45,49,36,50,44,44,73,61,29,40,17,30,90,65,25,47,42,37,19,24,34,33,37,45,35,58,54,45,69,24,36,57,56,35,51,46,44,35,40,61,33,55,19,31,43,49,48,38,29,27,31,26,53,32,22,41,59,34,35,41,42,24,35,45,33,23,42,30,36,49,41,26,50,20,59,30,23,44,28,58,33,38,49,42,49,35,26,31,36,25,35,40,42,52,51,19,43,57,35,38,23,21,60,26,49,47,40,42,33,63,53,38,44,43,26,56,44,34,38,34,22,23,44,32,38,47,39,32,40,34,57,71,45,54,28,45,37,57,39,27,44,44,19,55,45,44,26,32,43,34,33,40,20,23,58,25,50,42,35,48,24,44,52,59,27,38,51,24,58,51,33,46,53,30,41,34,46,35,40,46,35,43,61,29,21,36,23,90,24,40,37,22,25,46,47,48,36,20,55,45,65,45,42,37,56,31,52,45,53,47,35,65,39,37,39,39,25,30,46,33,70,36,36,36,40,36,28,39,49,63,61,48,55,20,49,27,32,33,47,34,25,26,35,56,40,28,65,52,24,21,35,38,55,27,33,32,34,46,48,23,48,67,36,43,56,54,39,21,31,22,45,33,55,35,53,20,45,27,46,55,31,64,38,17,25,33,31,28,41,61,20,64,28,39,53,25,27,22,24,27,57,30,40,25,39,27,33,65,52,27,32,45,18,27,44,30,28,61,49,33,50,53,33,22,45,26,47,52,34,37,57,40,23,54,38,32,42,22,40,26,46,28,46,27,50,75,42,51,43,44,60,47,73,27,50,35,31,45,57,34,25,28,38,19,32,35,24,34,28,46,38,35,29,41,40,17,46,38,24,28,60,42,48,37,32,55,30,22,36,52,46,53,29,41,30,46,45,59,40,59,36,40,62,47,20,48,29,36,43,33,27,60,34,30,17,29,37,43,30,29,24,27,69,30,29,17,61,32,21,27,44,45,34,28,34,34,24,41,50,68,35,47,35,26,20,52,71,25,39,27,22,57,34,24,41,32,55,49,23,43,40,41,23,59,20,21,44,31,50,54,28,34,46,28,23,29,35,41,90,35,20,31,23,60,28,32,44,34,27,49,47,34,27,31,24,24,25,38,57,24,41,56,44,25,27,22,59,34,30,24,82,35,32,28,60,31,28,24,37,36,39,22,46,28,43,60,33,22,39,33,39,37,63,46,28,24,48,35,62,42,60,50,39,43,35,39,30,32,18,51,33,46,40,34,29,34,53,52,64,42,44,41,53,25,56,38,62,28,31,24,65,40,31,35,42,25,22,56,51,36,49,28,57,33,49,48,38,46,56,27,18,29,21,49,47,40,43,62,63,51,62,48,32,36,30,27,41,61,36,29,45,18,28,27,38,33,25,23,40,36,41,49,47,28,37,44,53,36,77,21,23,47,17,48,43,34,23,61,63,37,41,35,51,43,22,56,26,45,49,23,69,41,40,25,52,50,48,27,63,47,37,28,41,67,28,26,46,29,39,21,25,44,37,50,50,58,36,21,39,50,55,64,40,42,43,28,46,34,21,65,20,38,22,44,41,38,48,24,50,50,68,31,22,18,25,45,57,33,24,22,51,46,50,64,51,28,39,47,40,23,31,40,48,50,18,53,54,81,40,44,69,17,38,27,52,24,38,21,26,28,37,49,28,57,60,20,38,51,37,50,35,46,18,32,39,30,37,38,56,35,56,17,46,61,25,50,33,33,38,51,27,42,37,48,41,33,30,50,54,37,59,65,57,33,41,50,18,61,67,41,29,27,37,22,26,27,26,51,45,60,35,24,44,24,52,21,67,25,34,39,26,19,35,38,45,37,37,21,46,52,23,48,26,40,90,44,29,45,43,17,49,27,53,19,28,22,48,76,58,22,52,31,50,62,26,38,44,50,36,39,57,32,24,54,52,84,35,56,30,35,44,60,40,50,58,41,28,53,71,25,40,27,19,24,38,18,35,45,24,63,40,21,20,49,47,48,61,23,35,18,28,62,28,57,56,33,19,29,30,44,22,40,25,30,46,50,42,41,32,32,47,50,33,47,64,34,37,34,50,60,44,48,37,51,52,43,23,66,42,71,33,42,23,28,31,28,37,19,51,25,49,48,46,34,35,30,65,34,34,39,28,53,34,21,53,54,39,29,43,20,39,36,62,39,45,19,43,32,34,66,37,41,63,55,35,28,37,63,43,30,54,36,78,77,39,37,34,36,40,38,28,22,43,22,48,40,68,57,23,22,23,41,58,33,28,68,33,44,38,49,41,33,22,41,66,38,43,40,32,35,41,33,36,33,53,27,26,35,46,53,18,34,37,61,59,81,39,37,44,37,41,55,60,22,23,45,46,48,39,29,37,27,28,54,17,18,33,49,41,40,38,47,36,47,47,56,36,34,56,26,49,71,38,35,40,21,49,54,20,31,19,33,23,42,40,32,63,54,32,29,26,39,57,54,57,23,27,42,31,50,37,26,24,24,47,33,20,57,36,25,46,39,29,46,61,53,36,75,44,49,34,31,47,38,42,29,38,32,20,41,46,18,60,36,30,40,56,42,26,48,35,41,55,42,45,29,42,30,43,61,34,31,32,44,21,55,31,41,26,21,28,64,65,34,58,63,50,46,30,28,51,47,30,34,41,57,28,22,38,34,45,37,21,49,24,47,43,48,64,39,50,50,54,17,33,39,33,59,54,31,54,42,57,40,19,38,39,43,32,38,34,35,21,43,43,64,19,34,35,32,26,25,55,29,38,26,34,18,58,29,49,21,28,29,40,36,34,57,24,18,39,39,20,54,40,27,43,37,49,33,21,45,33,25,71,41,30,50,31,68,45,38,78,31,39,36,30,34,30,47,62,44,20,45,55,38,49,49,20,34,46,58,37,42,43,21,45,55,24,70,27,34,57,37,52,32,42,50,36,42,39,51,38,18,50,82,33,37,24,18,47,48,34,50,57,60,39,29,30,21,51,45,36,44,39,45,28,48,34,35,28,18,32,19,23,38,36,45,30,46,27,48,31,58,51,72,61,42,30,20,65,37,33,61,27,39,44,73,66,53,43,57,59,46,37,41,39,30,59,39,63,34,38,42,56,17,33,36,37,28,59,30,32,30,53,54,26,34,34,58,30,53,23,24,36,47,34,44,45,24,65,23,54,50,40,24,42,50,58,47,37,37,22,29,45,49,44,63,26,58,40,56,55,37,31,42,57,53,38,70,50,50,28,33,47,48,64,37,55,55,22,50,47,30,37,73,50,34,24,66,22,69,23,70,54,34,38,33,60,32,34,46,30,29,51,31,23,35,33,40,27,38,45,66,30,48,77,41,35,66,52,50,22,19,45,61,45,36,40,38,52,29,33,40,34,46,44,40,24,76,20,33,31,19,29,30,48,42,37,54,42,42,29,43,44,28,20,49,36,40,45,42,57,23,23,39,34,43,25,40,52,36,33,24,42,35,50,40,22,57,51,47,25,33,33,28,20,34,36,61,37,27,54,40,18,51,40,64,26,31,39,22,31,68,35,25,23,35,17,19,53,74,28,40,30,20,80,51,17,37,58,57,37,23,33,41,57,23,31,41,36,44,36,42,48,22,38,31,32,35,21,26,33,30,52,38,28,57,51,50,32,32,37,69,33,37,36,47,63,21,17,56,90,33,42,66,34,38,40,22,46,29,19,52,31,37,28,25,31,29,69,50,54,34,23,66,61,24,30,49,56,25,49,58,40,54,18,32,34,53,47,41,56,42,57,40,20,47,40,48,39,46,72,53,47,40,34,41,41,20,42,54,37,59,69,36,21,68,20,39,33,46,47,65,23,34,20,28,52,29,31,42,22,21,43,43,33,17,54,55,51,65,39,56,37,44,49,36,59,36,25,45,43,30,33,34,21,31,48,18,43,37,24,24,38,48,43,67,53,22,45,43,20,43,21,36,64,46,23,27,29,54,27,34,38,35,23,42,46,27,22,49,23,24,34,25,33,75,28,62,40,40,45,48,34,42,39,52,25,40,34,52,41,36,21,31,20,68,35,32,64,25,31,51,67,42,29,31,21,67,41,51,28,21,28,29,39,21,43,44,28,31,18,34,25,34,43,40,19,49,20,36,23,21,25,26,22,41,24,38,38,38,34,53,45,31,47,27,33,38,45,23,61,32,36,24,60,64,36,57,23,29,34,50,50,17,19,43,59,39,18,39,44,32,48,46,57,46,27,29,51,35,29,30,47,41,38,38,52,46,56,59,19,31,22,63,33,53,38,39,59,45,31,61,66,26,60,32,43,26,55,31,35,58,56,22,46,26,42,19,51,42,43,38,36,29,42,61,55,23,48,71,39,38,38,26,32,44,40,43,31,57,25,27,41,31,41,25,29,22,62,65,37,33,38,42,27,41,47,55,30,40,30,22,37,57,58,35,26,46,26,22,30,58,17,30,51,26,24,44,43,35,51,41,62,45,32,46,29,33,23,63,60,31,28,31,65,31,43,39,47,45,30,30,49,51,32,19,33,25,49,30,23,24,52,32,36,39,30,49,42,35,32,22,25,17,44,52,42,34,27,37,26,36,17,30,32,58,44,59,42,45,20,35,50,56,48,32,48,33,50,52,55,36,44,34,33,35,29,56,33,24,48,65,29,42,44,73,32,43,21,30,54,53,51,33,24,36,58,38,72,60,27,38,37,28,38,29,28,40,26,43,25,29,60,51,36,35,48,78,72,24,58,36,48,47,35,17,41,37,35,56,23,79,28,59,41,32,67,43,38,35,65,24,37,34,42,31,40,32,47,32,46,40,36,33,23,21,53,20,53,58,51,31,35,23,39,51,18,32,35,61,23,52,19,33,35,22,20,19,47,59,36,27,24,22,37,36,32,47,40,52,39,43,35,43,34,50,30,44,52,35,36,36,25,26,20,70,26,48,48,52,25,58,45,37,23,31,64,32,45,39,25,30,28,34,27,35,37,32,50,25,21,41,47,48,39,62,22,46,51,40,31,30,37,40,60,27,62,38,49,21,18,25,36,52,19,63,63,62,51,50,45,44,51,42,43,53,29,41,44,27,35,49,33,35,31,54,57,30,45,41,25,51,42,28,53,46,47,26,30,48,22,32,33,31,34,17,29,27,32,34,54,53,38,43,25,39,57,47,24,32,25,28,74,44,27,61,25,20,33,38,43,38,19,56,38,18,35,32,62,38,27,26,21,19,20,25,28,36,23,25,24,30,62,45,34,65,27,53,54,30,59,42,20,73,47,44,40,39,49,44,50,39,55,53,31,21,30,47,37,47,49,18,35,51,21,30,25,31,44,52,30,29,36,54,59,55,70,32,44,22,25,31,33,54,42,49,30,39,24,39,23,41,32,57,60,38,21,24,36,29,23,44,32,41,57,62,29,35,52,38,41,23,31,31,46,57,32,27,46,43,42,90,25,48,48,27,32,22,31,28,40,39,54,59,25,49,55,39,28,35,44,51,55,45,32,45,35,55,59,23,47,34,51,40,24,20,44,26,23,40,49,32,51,62,31,47,44,46,32,48,57,33,42,17,63,31,34,35,20,33,36,47,30,34,33,67,26,27,24,44,31,28,48,56,22,37,38,30,32,56,66,20,25,42,61,27,31,36,51,49,43,45,49,44,67,58,36,61,23,35,23,39,47,27,29,21,24,39,51,47,49,20,24,54,25,26,24,18,45,26,47,54,21,68,21,28,50,48,17,59,23,50,53,52,38,43,41,22,39,45,40,47,53,53,33,46,29,31,29,42,50,39,74,40,75,36,61,42,60,30,32,51,53,48,68,63,32,32,40,50,38,30,63,25,59,49,36,37,48,43,41,54,66,59,62,29,28,27,42,60,58,27,50,43,39,21,24,59,36,35,33,40,26,37,45,56,50,38,29,39,26,52,52,46,38,62,31,31,39,60,25,42,45,22,50,21,60,51,30,46,43,57,48,39,68,33,22,18,28,32,56,36,34,39,27,56,33,49,44,30,27,25,30,38,38,62,55,47,50,36,23,37,38,27,41,27,51,44,33,28,32,43,52,24,19,47,53,31,30,56,49,38,19,39,38,47,58,27,59,24,42,53,35,26,33,29,43,35,27,21,39,52,48,53,33,37,47,36,41,67,38,47,39,34,36,24,50,40,34,44,44,45,24,34,58,39,22,32,45,23,35,52,55,23,27,21,32,37,61,53,43,60,25,47,25,27,23,59,66,46,17,27,32,53,22,47,68,39,62,21,34,30,40,26,57,36,49,58,32,59,37,40,47,56,21,22,43,60,73,62,28,45,19,30,23,22,45,42,62,61,34,56,32,24,26,58,23,54,35,30,23,42,21,37,32,22,52,35,34,53,23,58,38,26,60,28,29,41,63,31,41,51,59,24,47,21,38,24,38,19,32,63,36,41,57,48,25,48,74,54,39,24,19,28,28,49,50,58,41,27,24,64,35,48,46,24,57,32,41,55,41,45,33,38,40,24,37,42,22,45,29,24,28,18,59,53,51,59,42,32,38,30,27,65,23,32,28,44,29,57,41,34,44,57,37,35,51,27,28,37,18,40,54,31,35,40,35,25,58,39,69,23,48,41,74,43,18,24,54,60,28,41,62,27,63,33,34,20,51,21,50,43,27,20,47,69,48,58,33,56,40,28,42,23,39,39,33,55,31,35,30,57,50,51,47,37,58,34,19,42,40,39,36,37,60,56,51,19,58,39,32,17,70,30,51,19,47,41,36,30,26,41,25,62,23,25,48,20,19,31,36,40,30,28,57,18,54,32,52,45,18,37,52,59,27,43,56,44,37,28,58,29,37,35,36,52,30,22,55,33,46,28,36,43,39,56,24,40,45,36,50,31,63,41,48,24,34,26,27,55,38,51,56,39,50,34,44,21,37,34,34,38,19,46,38,83,34,38,38,25,38,52,33,34,17,22,22,46,61,32,50,51,20,38,44,28,53,42,47,38,34,43,28,46,66,54,38,31,50,30,45,28,53,33,34,41,67,41,30,54,33,20,20,17,21,25,38,38,48,25,57,39,21,41,24,57,51,32,28,40,21,45,55,42,49,38,50,47,39,42,50,37,56,33,35,44,57,43,33,18,47,59,35,49,26,25,50,32,31,33,62,55,43,58,48,21,39,25,27,41,59,23,54,39,57,31,35,42,31,46,38,51,27,33,22,72,25,58,23,23,51,69,52,34,46,44,33,22,23,75,24,45,31,49,61,44,79,44,24,46,25,48,25,19,26,33,36,33,34,33,62,34,25,40,35,48,63,36,39,32,19,30,35,23,33,25,64,59,49,35,35,27,28,21,44,39,40,70,34,66,40,61,59,47,33,18,25,22,40,68,73,45,23,40,39,23,51,53,24,53,49,34,37,40,36,42,59,36,37,31,46,27,29,39,62,60,39,22,18,44,21,49,52,39,23,54,21,38,51,34,26,41,35,27,65,31,45,37,47,47,33,40,38,21,40,33,29,41,20,34,78,46,40,59,23,23,47,31,42,29,21,55,44,61,38,50,53,37,49,30,59,43,29,49,36,26,41,55,20,37,51,28,21,30,44,45,42,33,27,41,41,33,33,52,64,42,29,38,24,43,38,28,44,81,55,31,27,23,26,33,44,18,43,37,33,45,32,47,44,22,17,64,30,51,26,50,18,48,37,51,69,56,39,41,45,23,29,34,46,35,48,23,17,44,26,49,42,44,49,42,30,36,23,32,40,28,32,21,71,31,24,69,35,35,42,51,48,49,53,43,53,35,48,37,30,41,24,34,46,63,40,33,36,25,54,50,58,17,40,27,22,80,52,40,62,32,58,37,29,25,26,50,30,30,36,37,34,27,39,41,34,41,26,38,27,21,41,42,25,43,23,40,23,37,58,36,36,62,34,23,42,44,42,21,31,42,26,62,28,34,51,44,47,35,46,46,46,66,26,29,30,44,54,54,51,32,57,34,40,50,32,19,17,37,32,20,39,18,42,43,42,52,51,77,38,20,23,41,47,36,42,22,29,42,24,41,29,36,78,24,36,49,21,30,37,20,46,24,32,36,60,57,50,56,61,45,31,60,26,28,45,58,49,22,23,40,21,43,17,37,54,24,35,23,50,34,36,33,32,22,53,36,22,23,40,45,61,27,25,49,51,44,35,28,37,54,45,27,49,51,27,36,59,50,43,21,33,29,42,28,48,19,36,41,32,49,38,29,52,22,44,43,41,63,40,40,57,54,20,29,22,44,45,52,57,51,23,22,61,30,18,23,57,58,59,27,50,22,21,50,44,53,41,60,58,27,33,24,41,21,39,50,20,46,49,56,41,41,42,58,43,23,59,50,52,37,39,54,52,22,19,65,32,25,34,24,28,23,57,64,28,28,34,38,44,18,36,73,24,29,33,21,36,26,32,59,29,59,58,47,31,42,19,66,41,30,75,19,60,24,25,37,18,24,33,57,50,27,26,47,39,49,28,50,28,30,47,39,37,50,37,26,55,46,32,26,32,38,70,32,49,25,27,43,38,42,33,30,29,30,26,43,38,48,57,42,31,32,28,63,32,20,36,24,21,27,17,29,39,42,22,40,35,58,45,29,21,51,38,37,43,37,28,35,35,29,41,37,20,44,23,46,32,21,32,41,51,69,39,43,51,43,32,44,40,21,23,19,61,47,39,25,33,54,28,33,63,31,70,50,39,17,62,61,18,39,24,38,38,17,28,52,61,69,26,34,43,52,59,53,55,46,39,78,43,36,36,46,24,41,68,38,47,41,29,26,36,40,21,53,34,43,64,25,47,48,57,37,51,49,52,64,24,39,30,30,58,32,18,71,40,38,47,59,44,50,33,43,37,40,48,22,57,40,42,56,25,52,55,47,31,42,38,29,26,37,79,61,60,51,34,35,27,36,27,40,43,25,41,30,26,52,33,35,62,50,63,40,61,40,41,28,35,31,34,57,38,37,26,26,27,33,65,57,18,44,37,27,59,19,36,68,37,23,22,49,38,46,40,42,48,50,17,21,32,28,62,23,36,39,21,39,33,37,22,69,18,65,28,35,33,26,38,33,44,42,44,44,41,41,62,27,38,24,24,22,22,51,31,42,49,37,66,52,23,67,42,39,54,46,21,27,39,23,36,24,35,37,41,42,41,31,26,37,51,26,40,25,38,43,44,21,36,43,54,57,35,44,36,58,54,37,60,36,29,23,38,33,49,57,57,37,43,39,31,31,36,57,30,28,22,31,33,20,25,49,45,36,23,47,41,29,64,55,19,18,50,44,18,34,40,35,26,44,21,52,57,27,66,41,58,50,31,32,37,26,33,26,48,40,62,37,18,45,35,35,34,19,53,49,38,39,29,52,20,30,58,24,27,28,57,50,27,27,68,53,32,53,40,60,34,37,25,46,21,18,59,33,44,55,56,51,43,22,43,30,30,46,27,26,36,25,46,66,43,36,46,42,39,36,60,37,26,17,54,63,22,27,33,50,31,53,34,27,80,40,53,58,38,33,48,31,52,24,45,54,39,40,28,21,36,55,71,47,17,32,46,39,17,33,72,65,36,32,26,22,47,31,28,34,44,27,50,20,24,29,42,20,72,45,30,33,43,56,25,31,36,37,58,55,46,44,28,42,41,39,64,20,45,31,36,33,36,40,44,31,26,25,35,29,39,44,37,79,36,44,23,46,58,36,36,21,49,29,63,65,42,51,41,18,33,46,40,45,30,56,24,71,26,34,48,30,51,32,52,52,33,67,31,31,28,36,58,56,35,37,26,51,31,37,53,54,30,42,62,31,22,45,34,64,42,60,44,67,53,37,28,55,74,27,44,23,41,45,29,33,61,22,28,20,28,45,34,31,50,40,24,45,80,23,46,38,50,90,55,37,38,46,50,54,40,30,50,28,34,55,20,21,43,33,45,32,60,41,21,33,43,27,18,48,37,58,57,24,26,43,34,51,22,26,45,23,62,42,39,43,33,68,30,32,25,38,62,20,26,23,29,24,44,19,47,22,50,72,29,47,24,22,30,74,70,27,30,29,18,31,39,31,33,39,29,34,35,33,43,66,51,32,19,54,45,49,28,39,27,24,45,19,46,50,47,69,43,29,43,24,24,30,39,43,34,41,24,24,33,37,39,36,42,31,57,67,23,35,43,39,36,25,54,24,50,70,61,51,38,28,66,57,44,42,74,34,17,43,54,23,34,30,32,40,28,44,28,37,56,27,37,66,19,54,47,47,48,57,50,45,55,25,25,39,27,43,33,25,17,20,19,71,31,26,52,38,25,48,53,30,44,32,59,65,31,53,44,36,37,63,29,52,37,31,44,18,30,26,68,31,48,80,26,63,19,42,64,32,50,73,53,58,37,24,53,28,45,26,29,43,31,56,25,60,59,46,28,33,24,29,23,20,18,31,33,63,42,20,35,62,19,47,36,48,48,58,59,42,33,23,21,32,31,23,17,35,39,45,25,29,56,34,64,27,47,36,44,45,31,45,33,25,70,22,30,46,43,59,51,47,36,73,34,51,57,19,45,55,60,19,35,34,26,27,37,73,46,36,54,49,68,34,30,28,28,25,26,40,36,90,27,37,28,45,23,69,58,22,58,53,44,31,48,17,20,19,29,25,54,26,25,31,48,22,18,34,30,31,21,41,45,30,21,36,50,47,36,32,30,51,55,43,37,39,35,59,26,68,46,21,34,31,56,41,46,43,35,44,58,44,57,24,33,33,50,35,36,22,30,39,34,39,19,29,66,38,44,34,29,22,33,73,35,41,25,47,37,26,53,37,41,53,23,28,53,32,52,31,37,39,42,49,43,43,31,27,54,37,53,26,59,36,32,53,30,38,44,68,41,43,48,41,52,31,54,31,47,40,48,25,59,66,63,43,21,44,40,50,18,51,29,50,50,43,41,58,50,37,21,47,45,34,39,19,45,34,27,32,24,39,62,37,44,53,25,44,24,61,33,48,26,68,25,58,45,24,35,51,47,42,39,20,65,44,33,25,38,31,33,29,23,37,52,27,47,61,32,21,63,34,21,26,36,39,52,44,60,31,30,39,28,36,27,43,19,37,31,53,19,56,43,45,28,46,62,23,47,51,61,76,24,31,43,41,21,40,25,42,35,20,23,40,41,25,45,25,44,49,44,19,60,46,42,58,58,69,19,28,30,36,45,45,53,27,22,41,53,58,32,32,27,25,36,23,44,19,42,60,41,67,36,28,83,17,27,35,51,51,82,53,19,66,27,35,17,27,27,64,26,40,40,25,30,52,28,45,17,23,41,19,69,28,37,57,30,25,33,36,28,35,35,20,69,46,41,50,23,40,54,53,17,22,30,40,36,25,29,29,60,37,67,31,26,64,27,32,59,44,61,32,48,50,44,42,70,65,39,32,43,31,37,49,29,35,37,42,21,23,29,45,24,38,39,45,60,41,48,33,48,26,55,35,52,72,62,26,39,41,59,37,38,33,67,39,32,43,23,32,48,43,50,63,55,32,27,44,45,17,53,58,55,44,40,58,38,31,31,30,38,58,30,39,21,50,47,20,20,33,23,34,25,48,26,25,38,19,19,24,59,67,49,35,44,58,22,21,50,44,36,30,52,35,48,23,37,44,44,36,26,31,23,29,52,55,31,55,48,45,17,22,36,56,43,41,36,59,46,46,53,47,47,29,90,63,23,76,56,58,45,25,25,29,23,41,48,29,27,55,30,50,22,30,29,56,63,20,31,21,47,38,30,39,27,43,32,20,32,37,45,37,24,28,23,54,40,38,71,52,34,43,36,29,51,28,54,47,47,47,60,55,53,28,27,64,41,28,61,20,28,36,44,55,44,35,40,25,33,18,46,63,40,59,23,64,36,17,36,53,20,34,39,37,51,38,28,25,50,50,43,27,32,44,53,21,26,68,31,32,33,49,24,30,49,18,51,52,52,23,23,36,19,53,17,33,40,56,29,22,28,31,38,39,27,56,80,52,26,41,24,44,54,33,18,41,22,66,70,24,35,38,39,33,37,38,38,49,38,22,40,20,24,58,43,42,27,37,48,50,36,41,90,23,35,32,33,32,31,32,40,27,34,29,52,20,18,63,58,59,49,17,20,36,55,35,30,47,29,32,55,47,50,42,32,60,53,58,45,31,32,26,47,34,42,38,49,35,61,45,31,52,20,64,20,24,29,34,55,42,37,47,20,53,24,23,29,55,25,23,34,39,21,45,33,26,52,34,24,77,34,53,31,50,32,37,33,32,30,27,21,46,60,31,26,25,23,53,45,70,55,40,17,18,34,28,23,32,56,36,26,28,31,23,58,55,54,33,28,40,33,27,55,53,23,66,34,39,59,61,30,22,50,59,28,48,41,47,29,30,19,59,19,45,37,34,30,37,45,34,27,33,37,49,37,22,44,30,55,36,29,19,45,38,42,20,43,21,36,34,54,30,63,36,25,55,44,20,47,71,41,35,57,56,61,32,52,48,35,34,27,47,65,18,49,48,26,30,68,23,65,32,44,24,58,32,54,35,24,50,38,57,38,49,56,22,23,33,62,24,26,46,65,65,46,43,35,55,45,56,59,56,21,26,48,37,63,27,20,26,21,51,28,35,65,54,39,70,52,64,43,44,32,49,44,28,46,37,62,37,35,44,41,36,48,39,44,25,26,23,55,45,42,29,48,36,52,46,34,42,64,36,52,44,53,36,27,52,60,41,64,25,63,54,21,27,63,22,54,60,19,44,22,39,49,41,35,25,25,37,53,22,45,33,45,43,40,41,25,52,45,26,24,43,35,37,17,48,26,35,54,22,20,21,21,36,65,48,37,20,55,80,59,52,39,56,49,33,28,49,51,44,26,52,24,41,36,19,43,48,50,21,33,50,49,34,70,39,40,46,21,47,19,41,41,28,20,24,33,44,47,24,58,41,38,33,38,45,65,33,34,65,24,26,55,43,22,21,48,26,60,39,45,76,34,54,44,23,37,41,37,63,35,47,41,26,36,47,32,37,40,56,50,28,28,44,47,31,61,26,29,40,33,38,50,66,36,55,46,36,29,37,27,26,23,60,46,21,48,70,58,42,28,65,35,82,48,44,42,54,22,43,46,59,37,70,57,63,44,30,23,44,34,37,42,34,25,36,60,64,33,57,39,25,58,57,40,43,24,34,64,55,26,58,43,43,46,51,43,37,32,65,28,37,40,35,62,49,47,37,36,36,44,51,25,28,46,59,28,25,27,29,43,55,36,37,23,34,41,35,57,28,52,31,21,50,43,33,23,35,18,33,21,38,33,24,24,43,43,62,25,22,73,26,24,18,35,28,38,68,32,42,33,54,63,67,25,35,54,35,18,30,29,40,46,49,34,42,31,31,18,30,27,28,26,41,32,23,35,42,34,46,34,25,45,46,23,37,39,38,35,49,44,19,42,29,48,56,32,57,55,48,46,66,26,58,62,31,25,38,20,40,20,35,35,18,37,58,55,41,50,48,42,45,43,37,43,60,27,36,47,33,19,34,60,33,20,24,31,55,49,33,24,46,50,22,32,46,62,39,37,41,40,42,20,55,29,37,61,26,49,18,30,24,37,36,40,44,43,17,43,43,40,31,25,30,55,55,29,28,30,33,23,22,62,49,31,27,27,26,30,26,50,31,24,29,26,51,32,47,30,49,40,62,43,76,40,35,54,55,21,20,26,28,56,39,44,55,54,30,34,34,23,27,81,57,33,47,42,39,58,20,19,39,51,50,36,50,46,24,26,56,50,22,51,58,45,55,45,42,44,60,27,25,43,44,47,37,46,53,79,44,54,47,25,37,35,46,45,19,41,33,22,29,34,23,20,59,44,55,42,47,42,45,23,60,23,52,49,20,30,45,30,22,25,52,18,58,30,59,33,22,25,40,37,32,52,36,57,80,40,36,35,29,27,49,26,47,19,34,23,25,21,25,63,39,30,36,47,27,52,50,21,47,54,32,61,21,29,38,59,43,19,42,33,51,28,31,38,21,22,35,53,36,33,51,31,26,32,43,49,45,41,46,44,25,30,54,34,34,30,29,60,36,57,63,29,34,27,31,64,44,35,21,64,42,31,42,23,20,56,19,45,41,29,48,38,35,59,27,43,70,30,24,76,25,20,54,22,41,35,23,54,17,33,38,54,32,61,22,34,20,53,68,28,54,21,25,30,33,45,67,42,69,34,28,31,32,76,43,48,39,59,40,63,22,36,57,58,25,59,36,29,49,42,43,46,43,26,58,43,53,74,45,42,24,63,37,28,31,46,44,45,52,40,26,55,27,71,35,17,26,51,33,41,50,53,32,58,47,51,42,28,50,58,47,64,51,47,68,39,44,46,40,19,65,40,61,30,61,33,32,32,37,53,51,29,48,69,45,60,38,19,18,33,37,50,19,47,21,36,62,65,46,35,50,33,53,47,62,35,23,23,34,18,38,41,36,30,26,45,51,34,42,26,20,43,27,46,39,33,42,23,27,43,23,35,30,59,65,28,79,54,27,30,18,28,46,77,41,48,29,41,34,49,49,30,28,62,41,20,61,30,44,25,35,22,26,26,44,32,66,19,54,37,38,37,22,34,35,42,34,38,62,35,47,37,60,31,74,63,30,44,40,35,50,33,59,17,47,51,26,55,54,52,59,55,54,31,19,31,32,31,32,46,41,39,23,53,45,24,53,38,26,18,26,48,58,40,33,74,49,25,57,29,28,59,30,48,19,31,28,41,46,33,34,47,81,45,35,46,39,23,37,49,25,32,34,40,78,31,61,22,29,37,52,18,37,46,25,47,55,18,25,51,47,37,27,52,41,55,24,41,37,48,40,65,44,32,41,28,44,31,34,20,33,52,39,35,37,44,28,35,19,30,29,27,37,44,34,41,34,23,24,24,53,33,24,45,31,38,25,29,48,36,62,60,41,43,30,47,50,41,23,30,61,71,38,27,45,25,52,52,36,69,36,31,38,24,37,69,39,38,34,61,35,29,41,42,51,55,23,24,48,28,34,25,48,32,31,22,27,30,50,42,37,26,33,57,45,25,56,25,23,35,28,41,26,22,30,42,49,27,41,25,23,68,17,45,30,42,26,33,45,66,33,38,72,39,41,22,49,24,25,26,50,41,60,77,22,17,31,23,27,61,55,38,25,66,32,43,18,20,20,33,40,33,30,40,44,30,45,22,25,63,55,22,41,21,39,18,60,22,34,27,37,25,33,34,51,33,57,25,56,34,36,46,55,88,60,56,36,67,35,34,23,19,25,53,31,33,33,45,20,36,36,51,45,45,24,63,35,44,52,27,35,51,49,35,37,38,58,57,31,17,40,65,31,71,25,40,40,38,38,42,23,56,36,22,44,19,37,54,18,22,63,26,20,39,28,42,40,67,18,36,34,24,30,49,46,34,23,32,22,48,27,43,55,31,34,34,37,21,35,40,23,30,57,34,24,41,38,43,35,42,42,48,37,35,59,57,26,26,19,46,22,72,36,76,37,42,44,53,28,69,20,33,47,50,22,20,42,23,24,41,35,40,25,34,23,30,42,39,52,42,25,27,30,23,47,58,58,35,31,46,43,52,39,34,58,26,29,67,62,54,36,18,25,56,59,36,27,30,39,31,28,38,44,40,31,43,30,36,19,36,42,38,33,52,30,18,24,17,66,55,44,23,29,54,30,59,37,45,55,49,67,49,30,33,25,28,23,37,32,57,36,30,27,21,22,43,20,30,57,44,42,26,42,30,59,21,56,21,48,35,43,17,41,35,22,44,39,59,34,31,59,46,27,18,46,55,48,44,43,27,34,40,55,26,54,22,33,42,19,45,53,25,90,45,47,38,35,37,53,56,18,44,26,38,46,41,37,38,50,34,23,50,45,17,50,31,45,25,20,42,56,56,33,50,39,35,40,59,42,59,54,38,53,54,41,22,45,29,42,42,19,53,24,37,26,25,52,50,40,32,37,47,47,61,47,56,30,36,42,45,28,31,32,29,26,68,32,21,33,26,52,31,29,20,40,43,52,24,20,38,32,49,31,46,46,37,44,28,19,31,47,37,22,46,51,50,38,40,58,36,49,46,53,60,28,31,45,34,27,51,55,43,48,42,51,32,34,31,71,17,34,46,33,41,59,50,46,45,31,28,32,36,39,23,44,37,43,37,26,34,51,67,40,33,48,24,24,40,42,31,53,32,34,41,25,40,17,82,26,37,25,65,39,36,37,23,34,63,49,28,45,40,45,20,41,49,37,31,21,20,46,48,50,41,54,44,50,34,46,35,38,37,30,21,40,56,48,52,60,45,30,31,43,36,42,62,45,47,43,24,25,21,29,21,41,55,42,63,21,45,43,53,34,46,53,33,45,48,55,46,21,53,61,58,36,41,30,50,51,41,47,51,60,20,42,19,34,38,40,39,38,23,20,49,47,43,56,60,29,39,38,50,64,35,42,51,21,53,34,37,18,57,47,27,23,36,40,59,39,32,51,31,53,28,19,40,21,39,40,36,32,50,18,19,53,40,31,47,35,51,54,76,33,19,49,26,49,24,48,36,31,53,43,31,36,47,52,24,21,29,34,61,20,47,40,53,26,39,17,46,34,45,41,18,38,28,49,29,30,25,27,25,26,49,44,37,39,65,40,31,71,25,36,56,34,39,44,60,65,44,53,28,28,53,26,34,50,47,27,55,35,39,30,39,25,57,32,42,39,46,53,36,34,25,45,31,21,33,29,20,21,43,33,61,32,24,58,42,56,65,57,36,42,18,18,28,19,37,23,60,29,34,36,47,61,57,33,54,40,40,35,32,49,58,26,35,25,43,32,45,48,31,28,22,38,23,24,24,84,48,46,31,58,35,36,36,47,57,30,30,23,41,52,27,59,69,24,19,60,39,51,26,23,57,20,37,46,50,25,44,41,30,38,54,37,46,38,71,35,62,56,30,35,21,46,33,36,42,35,25,47,44,51,53,47,20,28,58,35,30,56,45,62,23,36,40,30,41,59,27,19,41,27,38,30,48,39,40,27,59,34,47,64,29,38,42,49,53,35,70,38,34,24,59,33,63,18,33,19,27,40,38,20,35,40,34,23,26,53,34,39,25,18,65,64,68,35,43,37,45,47,34,44,25,35,42,23,50,38,46,38,42,48,39,34,23,32,45,21,36,40,68,40,27,57,54,30,17,35,56,20,38,24,31,44,41,23,37,46,43,31,40,64,31,48,37,28,42,60,46,22,44,36,26,21,41,19,40,19,40,37,33,34,44,72,22,59,50,40,22,42,37,19,43,50,47,31,33,27,59,39,29,24,25,52,51,30,23,37,25,49,30,20,35,24,32,29,25,50,24,49,37,20,49,55,23,43,21,52,30,42,50,49,45,36,20,61,72,54,44,19,20,58,20,62,67,37,25,41,63,60,31,29,51,45,27,18,66,36,27,22,41,20,52,44,49,71,20,59,41,55,45,43,19,63,39,34,31,21,45,59,58,29,39,57,50,44,56,26,33,36,45,23,23,35,45,56,44,21,31,20,44,71,21,29,46,36,70,34,26,24,28,24,76,19,45,24,37,24,40,20,23,53,52,28,31,22,38,45,50,53,24,33,57,30,31,39,39,34,45,19,25,42,66,63,43,26,39,47,35,25,43,31,45,50,40,55,64,51,30,35,55,34,43,24,30,33,32,50,18,45,22,45,41,24,22,28,23,49,37,70,83,69,71,42,21,29,27,32,31,55,48,58,49,23,40,59,37,39,43,40,48,48,37,19,56,45,60,53,37,31,29,39,31,42,24,62,47,76,46,29,26,31,41,29,60,72,41,25,24,59,17,59,69,25,50,43,29,32,17,50,43,65,70,53,37,21,20,47,26,31,45,55,49,33,30,19,48,41,29,47,53,43,56,28,44,30,31,64,45,40,19,54,61,62,33,54,24,42,51,47,21,43,53,44,31,48,41,35,54,40,23,34,27,32,72,43,53,18,38,29,70,28,55,50,42,22,42,45,55,57,25,43,34,48,38,17,70,45,56,64,23,42,23,26,56,41,47,21,29,32,69,19,25,56,24,40,37,36,41,36,42,29,23,27,32,33,21,39,50,32,40,58,56,30,25,33,41,50,44,27,27,58,25,39,55,44,61,32,35,28,57,32,43,60,48,36,41,33,72,34,27,75,51,47,31,36,18,34,51,20,59,31,68,30,34,49,38,60,51,24,21,48,40,30,30,56,48,42,23,43,22,41,71,42,31,69,58,31,35,62,27,17,35,51,64,54,43,32,36,26,37,26,50,47,37,29,63,50,42,32,42,41,25,65,32,32,24,32,77,26,20,57,40,50,19,54,51,51,38,68,32,29,42,27,61,49,41,46,28,39,50,57,79,35,35,30,26,48,22,47,36,44,34,37,29,24,29,59,27,50,25,36,62,38,19,49,60,28,18,43,65,23,20,32,37,29,59,22,31,59,56,37,58,41,29,38,36,19,42,29,37,39,28,36,44,60,59,31,40,33,41,62,43,55,55,44,23,63,49,25,44,17,42,26,24,30,19,52,17,27,37,40,49,43,32,30,25,42,60,46,39,28,26,59,64,90,36,30,47,32,23,62,62,35,27,18,63,61,50,36,21,38,30,22,39,42,47,42,29,35,28,49,60,26,25,56,35,54,28,31,52,19,23,23,52,33,58,61,39,38,59,41,19,22,37,31,18,28,49,38,19,43,28,50,46,20,45,46,66,25,55,48,47,55,41,47,40,41,52,62,46,21,35,21,35,46,70,35,34,21,29,41,51,47,42,46,23,49,41,50,27,29,54,32,36,50,32,38,31,38,50,18,59,56,34,32,29,29,42,32,52,29,39,39,60,51,24,20,58,21,28,40,45,44,26,29,23,45,51,22,67,28,33,37,35,25,21,32,65,57,31,37,26,44,66,31,53,61,43,38,26,27,56,36,29,28,61,66,31,67,26,21,33,59,82,29,57,41,55,27,33,43,68,41,30,31,41,66,31,32,51,31,20,40,53,56,51,38,44,49,47,53,41,19,61,45,29,41,38,37,55,62,48,43,25,49,55,40,34,39,43,27,67,21,17,31,51,29,28,32,31,26,51,62,54,24,27,18,44,50,38,39,27,35,51,60,62,38,37,32,28,83,42,46,44,60,52,73,26,20,50,23,35,43,50,58,55,29,27,25,31,42,52,46,52,25,54,46,28,24,60,35,47,40,38,54,39,60,49,42,27,51,51,40,61,35,31,43,20,35,30,42,51,39,69,63,21,24,55,42,52,29,32,47,56,40,36,45,38,60,70,33,37,50,68,32,30,18,23,59,48,53,18,50,63,38,60,30,39,55,17,46,32,34,50,32,26,43,32,48,28,43,30,34,34,41,42,81,40,49,46,28,35,29,52,45,43,21,35,30,17,32,32,30,28,62,46,26,41,57,52,35,19,49,47,30,32,22,47,41,36,32,37,36,64,17,42,37,56,52,37,48,33,39,34,23,49,18,20,54,19,53,28,35,64,42,27,36,23,46,24,62,19,20,47,19,35,50,27,28,29,25,49,31,56,45,72,46,22,41,46,32,52,48,25,60,38,37,40,48,40,68,35,33,20,36,34,19,34,37,47,23,33,60,17,67,57,17,53,25,24,48,51,38,51,42,39,20,21,26,54,32,59,54,33,45,58,27,20,32,45,43,54,46,58,50,74,30,36,36,37,29,17,40,42,32,33,43,47,73,37,19,32,44,44,26,29,21,49,24,26,25,30,52,30,20,31,44,46,58,41,72,26,45,48,23,18,25,54,41,23,17,42,61,21,44,48,46,37,28,19,63,20,28,48,38,45,38,55,63,21,62,46,35,37,45,19,61,61,45,38,40,34,19,46,44,53,68,62,62,45,29,33,34,65,22,30,44,52,53,41,37,55,36,19,24,30,17,28,34,35,24,20,55,20,38,26,58,23,53,35,41,31,57,25,23,48,37,43,20,40,35,43,40,54,32,49,52,41,19,21,20,46,39,34,19,27,53,46,23,39,30,53,33,39,37,17,43,46,17,60,63,42,30,33,43,30,37,49,35,27,33,41,38,45,18,28,36,21,44,33,61,29,45,29,21,49,35,47,37,37,26,18,55,47,53,38,19,28,40,35,35,41,38,29,46,40,29,39,31,27,27,57,36,51,39,43,38,51,43,44,40,47,26,29,30,57,37,39,45,59,28,47,29,20,32,48,50,31,23,35,50,39,52,44,43,39,33,31,28,35,44,40,55,30,38,75,17,79,20,25,27,19,42,28,47,50,48,33,45,24,24,29,24,63,36,58,42,39,39,28,51,45,45,38,38,39,52,31,22,41,29,42,52,46,21,74,29,44,23,38,33,23,37,25,36,18,25,29,53,33,27,24,54,36,24,39,44,24,52,46,50,34,40,58,26,22,36,36,60,49,50,34,46,46,69,32,40,61,56,47,29,37,44,51,24,40,34,56,40,57,26,53,39,56,17,29,33,47,43,64,40,21,31,32,66,27,18,52,31,28,37,35,42,18,46,54,54,27,41,41,36,37,31,46,24,25,55,64,65,46,45,32,34,35,26,24,45,44,57,41,20,32,22,21,40,59,45,45,22,26,21,26,31,36,66,34,80,40,19,42,59,26,42,19,34,49,36,53,21,39,52,59,50,31,41,45,22,50,48,24,54,26,53,51,54,32,46,60,56,57,29,49,28,45,17,54,31,54,55,37,34,38,51,41,24,56,33,31,27,32,35,43,19,23,40,59,43,43,26,68,41,32,25,20,22,33,26,68,32,30,67,53,27,29,38,48,18,28,34,62,46,20,29,55,20,54,27,43,17,48,34,66,43,56,58,54,45,29,43,33,45,25,25,69,37,20,27,55,39,34,51,67,40,58,45,51,43,36,58,33,60,24,42,73,59,28,19,58,21,47,30,41,43,44,28,33,30,46,21,26,27,50,28,61,30,40,36,28,49,61,41,57,49,59,41,59,20,45,25,45,58,39,34,22,44,22,27,46,56,36,35,29,19,38,40,52,36,22,32,22,42,27,52,43,47,61,39,47,19,37,31,69,22,29,24,42,30,27,35,22,25,30,25,70,50,34,57,43,60,46,26,22,53,43,24,24,52,32,35,38,34,40,40,50,32,18,45,46,73,64,23,30,35,31,40,46,42,23,21,49,53,32,20,17,25,58,24,48,20,60,20,52,46,37,44,63,22,31,17,21,67,46,29,27,35,43,62,28,59,37,27,62,46,28,46,54,34,67,62,30,19,49,27,36,52,56,53,31,41,63,39,31,35,54,55,31,58,31,38,17,48,73,25,53,46,30,29,29,58,62,45,38,36,42,47,42,37,54,48,29,37,55,51,41,23,38,62,70,45,38,50,40,30,28,62,40,53,42,26,51,49,43,42,26,35,53,20,31,28,46,41,30,66,37,46,51,33,30,47,24,54,32,45,46,37,48,58,69,38,46,19,24,38,19,28,38,21,48,30,24,32,56,38,19,61,28,50,43,28,31,35,63,31,22,67,50,31,17,41,53,44,24,35,44,56,45,69,64,43,37,59,50,33,56,28,42,34,19,68,28,22,17,25,25,47,44,46,38,23,33,52,35,67,60,38,29,65,29,43,42,32,41,21,33,19,19,35,45,25,39,26,60,22,34,39,56,23,39,24,51,47,25,34,32,49,43,71,59,25,32,57,17,41,62,39,30,37,40,38,24,33,48,51,48,43,34,41,51,25,20,61,26,46,39,18,29,39,47,40,43,32,19,44,46,53,43,35,48,28,36,28,21,34,36,31,44,49,40,64,54,27,68,64,36,21,22,61,24,49,66,29,57,29,47,59,42,29,25,44,60,46,44,18,33,63,52,17,45,31,35,31,48,28,42,43,30,24,46,50,22,18,53,40,36,32,39,46,26,19,28,44,38,42,28,36,47,46,26,67,54,26,27,27,52,21,27,46,42,79,43,51,45,65,66,36,28,39,43,29,32,60,34,29,41,43,19,29,40,53,27,44,25,38,48,50,50,74,34,19,39,27,46,18,50,42,42,37,27,31,56,35,28,38,30,35,45,35,41,60,18,19,24,56,28,29,35,55,18,53,26,55,45,28,32,27,41,23,22,36,65,59,33,31,50,49,34,63,33,51,31,48,47,54,23,34,34,31,41,27,23,59,56,35,59,47,23,43,46,33,36,35,30,39,37,22,25,64,33,59,41,60,19,23,57,36,17,24,32,27,21,22,29,22,37,38,34,45,36,20,34,32,58,64,28,21,25,41,41,27,54,29,36,41,60,34,60,39,59,34,36,54,48,19,40,25,39,34,35,36,59,33,22,48,45,46,35,51,31,39,26,25,61,52,52,50,76,25,66,27,41,43,31,33,28,41,44,47,24,34,27,35,59,39,41,32,43,67,20,35,38,42,44,28,74,50,43,29,61,27,47,33,45,42,35,20,40,60,49,38,38,35,40,40,36,24,62,54,36,31,44,33,62,55,26,67,25,30,22,19,61,53,42,55,35,36,45,48,40,31,26,31,55,38,21,27,35,27,18,66,33,28,34,30,59,42,60,53,41,49,36,34,40,48,51,22,34,34,56,49,26,48,35,29,47,53,50,26,70,55,27,31,43,24,41,52,59,58,65,24,51,31,51,24,30,22,30,49,42,57,26,31,38,44,42,23,31,35,30,35,23,63,20,50,49,53,54,40,28,24,19,42,54,42,30,33,39,27,20,62,17,69,34,34,20,27,40,53,44,46,17,35,49,26,35,29,54,36,25,18,22,52,49,42,37,39,32,53,30,38,23,39,37,41,47,47,30,29,42,51,34,44,27,39,46,32,23,47,66,26,52,48,37,50,39,38,53,28,42,40,20,32,23,25,41,42,18,29,36,34,34,48,36,67,53,53,45,25,47,31,38,23,78,54,30,46,66,23,25,32,25,51,24,26,41,39,51,27,33,67,41,29,28,25,21,24,46,35,35,50,45,39,33,50,26,49,49,33,72,32,44,22,17,49,26,19,38,34,51,48,63,38,54,41,65,36,42,38,20,37,35,51,19,37,29,33,60,22,22,35,30,43,35,61,42,36,24,47,51,49,63,58,36,19,29,43,60,21,33,37,17,37,28,57,45,32,36,34,27,47,46,45,32,59,45,76,32,38,53,41,27,44,35,42,40,45,36,57,62,29,38,29,18,19,34,25,34,37,37,18,24,29,49,71,35,55,37,49,32,63,39,42,30,48,57,28,43,46,60,37,59,72,29,35,24,24,42,58,39,25,69,36,34,45,27,28,44,36,39,30,34,50,30,43,26,50,45,46,17,56,23,33,52,34,26,35,58,32,28,20,17,56,45,54,58,39,19,37,49,28,23,19,37,41,56,44,30,38,32,57,35,48,32,51,50,53,59,23,44,23,44,30,28,38,39,28,33,59,26,28,48,41,49,37,61,26,66,66,46,62,35,26,31,52,61,39,47,33,17,26,25,39,33,46,48,40,24,55,56,25,36,26,37,36,23,25,37,39,35,33,31,49,34,40,50,55,49,40,51,30,70,43,22,38,52,32,21,27,31,49,61,46,37,56,37,26,52,40,63,30,50,41,39,52,30,23,39,37,40,47,20,35,62,42,32,53,28,26,47,70,41,43,40,64,40,43,50,32,42,41,20,45,42,24,28,36,50,34,35,61,27,18,31,31,35,51,43,18,55,49,61,64,63,18,55,34,29,52,57,38,51,40,41,28,59,31,30,54,33,32,27,40,34,34,28,65,61,35,25,48,54,47,30,67,26,61,54,50,36,50,28,58,27,62,36,28,33,54,76,34,17,26,44,65,54,34,42,26,39,34,59,20,34,51,34,21,30,38,34,22,31,50,19,35,28,27,46,27,34,33,30,36,50,19,52,20,34,60,19,59,36,25,49,33,40,56,56,23,67,54,59,42,26,47,78,24,27,51,46,55,20,30,36,30,31,26,59,53,24,39,53,75,36,46,56,29,60,58,67,35,39,37,54,63,55,39,23,44,35,48,48,33,18,24,32,44,27,39,26,34,38,41,73,32,56,59,44,35,52,30,23,61,68,41,42,31,40,51,44,30,24,34,17,61,30,45,49,39,52,38,19,25,71,54,49,57,56,17,30,26,61,50,24,25,34,38,42,17,48,38,52,41,49,19,39,43,42,17,48,60,57,27,24,41,42,50,30,38,31,34,52,34,31,37,36,72,40,19,55,30,29,41,46,44,38,36,24,38,30,36,27,27,66,65,17,19,57,43,50,38,31,62,37,53,28,46,19,84,48,44,25,37,24,30,56,30,40,28,39,33,23,32,37,30,53,42,44,54,38,29,20,32,24,59,56,40,35,30,61,36,31,34,24,42,23,60,47,28,55,47,54,38,62,29,19,40,36,42,43,39,38,49,35,51,23,51,33,50,38,26,26,33,42,20,31,36,39,52,61,27,21,38,51,59,41,43,17,32,42,68,55,43,61,28,21,39,39,34,57,27,25,51,19,27,36,33,60,42,42,34,30,34,37,31,33,41,23,51,30,49,37,42,67,31,50,33,27,31,35,55,56,46,40,52,59,33,20,21,42,54,46,21,39,32,23,42,28,28,35,27,54,30,66,30,32,26,52,30,19,33,20,61,31,61,61,34,32,38,26,35,50,34,37,35,55,25,51,37,54,47,46,24,19,35,36,46,34,58,31,33,28,32,41,48,27,34,25,22,61,58,30,24,52,49,34,48,30,44,44,29,35,27,37,33,34,28,41,19,48,26,32,36,49,33,54,36,41,27,68,21,56,40,46,43,46,51,31,27,36,31,52,39,24,55,38,36,30,34,19,79,28,41,21,45,23,32,67,32,35,29,38,38,18,47,21,27,32,40,25,44,31,33,41,29,32,78,64,58,55,35,27,37,47,25,53,72,25,37,39,64,29,50,20,34,43,44,47,51,41,31,25,46,53,28,60,43,45,28,19,51,19,64,44,21,33,63,44,24,20,21,34,26,35,25,27,52,36,37,25,54,42,65,61,29,34,40,42,50,49,38,41,70,51,57,31,45,39,38,38,19,36,30,28,53,32,57,36,25,43,56,35,44,44,64,26,53,50,45,59,65,62,21,41,35,62,53,69,47,52,54,58,30,73,64,32,51,57,25,53,28,46,40,24,20,50,35,54,35,31,47,33,35,36,39,25,25,53,18,41,25,66,27,23,29,26,26,38,36,34,32,57,35,75,28,39,31,27,42,49,33,61,61,39,30,34,53,46,27,23,22,25,29,90,32,18,50,54,43,28,51,31,57,28,25,49,52,23,32,30,24,45,30,52,49,20,41,31,32,29,21,43,39,40,35,35,57,35,31,39,29,42,46,73,27,38,19,60,24,52,28,38,47,27,49,68,36,43,49,25,34,34,64,51,31,42,65,23,31,17,63,48,45,17,26,43,39,43,49,52,32,38,44,63,21,32,50,26,35,31,24,40,31,43,44,75,52,51,39,40,40,33,38,29,37,19,46,46,37,26,57,33,26,38,61,36,61,17,40,18,41,55,27,39,36,51,27,30,42,56,41,49,54,38,45,50,31,27,34,46,47,36,22,51,43,54,29,18,31,57,48,24,24,27,33,19,47,20,36,49,58,38,32,35,51,33,48,53,40,64,42,25,42,24,22,33,63,20,33,56,70,42,28,80,37,38,30,36,70,55,22,41,62,52,36,74,41,25,35,38,34,48,51,39,17,42,32,33,25,44,72,36,67,28,24,17,21,51,32,39,50,34,68,49,18,46,39,35,21,29,30,37,45,45,20,35,24,19,23,23,36,46,38,27,48,53,47,50,32,40,41,47,45,44,49,45,41,29,46,19,56,55,29,38,52,20,31,39,35,45,61,18,51,37,48,62,21,28,19,41,47,43,20,29,40,26,68,41,43,20,43,53,45,39,31,23,37,41,23,26,37,33,37,31,27,35,59,42,50,27,43,46,41,57,46,45,29,23,46,55,58,29,57,42,61,31,43,47,51,23,25,22,38,33,46,52,46,28,37,45,56,38,81,42,66,50,31,33,37,50,55,39,34,60,58,27,63,23,49,45,42,48,33,24,49,40,57,46,35,20,42,46,54,28,38,22,31,27,45,38,75,37,34,25,27,35,27,42,23,38,28,56,23,48,45,38,24,47,45,24,40,28,32,20,23,46,59,31,34,19,20,29,54,36,43,46,33,36,37,37,64,45,28,23,46,32,26,33,60,39,45,29,53,46,20,38,36,29,64,34,36,68,37,22,43,36,47,33,33,48,28,33,44,33,22,34,31,38,36,29,64,24,28,18,41,17,58,50,30,47,24,50,29,31,40,53,43,50,27,20,25,30,38,67,35,75,23,42,39,41,45,61,41,22,55,66,25,67,44,42,29,40,26,52,41,27,59,19,32,39,33,38,27,59,25,37,31,26,59,28,22,20,27,20,43,27,40,50,60,24,24,45,58,22,41,48,31,33,49,30,38,58,61,58,32,33,35,59,45,50,36,18,28,42,60,41,42,36,52,42,26,35,19,38,36,39,64,49,26,66,38,30,22,60,54,46,33,43,60,39,40,48,24,32,31,39,45,31,45,50,33,22,27,29,60,46,23,38,60,37,58,25,33,21,45,44,52,53,18,32,48,46,28,27,69,50,52,50,58,46,40,25,33,49,56,23,22,35,42,22,32,25,26,46,55,24,25,31,54,47,29,53,50,21,60,31,24,47,39,27,36,54,42,52,23,33,35,27,24,44,45,75,47,64,54,41,33,51,45,18,43,30,45,21,56,28,55,27,33,43,18,35,19,39,29,45,44,42,34,52,18,64,30,43,49,35,37,49,22,31,26,44,33,18,28,30,33,36,36,25,21,51,38,36,41,19,36,46,29,35,35,40,22,70,38,35,39,50,43,37,51,20,17,25,46,42,46,49,21,43,22,40,50,48,30,34,42,48,38,29,33,64,42,49,25,38,44,45,27,18,38,45,27,66,45,42,32,26,27,23,47,46,55,36,38,37,81,25,30,65,18,26,30,61,21,53,43,47,48,31,33,27,44,37,34,51,51,25,29,36,31,31,30,37,30,27,32,23,35,45,27,46,42,33,46,24,50,48,55,66,63,61,64,35,58,58,33,27,30,22,49,46,17,45,26,44,36,47,32,35,37,36,34,24,31,38,30,35,28,47,49,33,44,18,60,44,81,35,58,32,40,44,33,42,27,62,31,50,50,46,66,41,42,51,45,60,39,46,58,21,41,29,33,42,24,59,36,54,39,42,54,50,42,34,41,44,34,26,37,64,48,44,32,42,33,47,62,62,38,33,17,58,34,32,19,41,52,48,18,25,56,55,32,41,53,21,29,73,43,45,38,49,41,27,64,41,37,61,40,20,52,50,33,40,45,24,54,43,63,19,49,27,60,71,34,30,46,32,25,25,32,51,20,40,19,53,42,38,41,64,30,20,20,35,33,43,48,33,50,23,36,37,43,38,42,47,44,28,51,49,62,23,66,37,69,38,37,38,37,23,48,55,35,35,53,43,36,45,36,22,19,33,64,47,45,45,27,41,20,54,50,31,49,44,49,28,43,25,30,48,61,41,36,24,27,18,23,43,28,47,51,35,38,47,31,45,46,70,46,20,43,37,40,24,31,59,40,30,28,25,21,67,54,36,29,55,19,40,71,51,27,20,28,36,31,19,38,36,17,35,32,42,36,58,39,51,26,21,45,26,68,60,65,34,28,57,37,34,57,55,51,56,32,18,48,46,41,33,33,47,42,31,51,63,52,30,34,31,38,43,18,36,43,26,54,65,41,50,36,48,35,37,32,29,41,38,54,36,33,69,50,40,31,28,43,62,21,37,74,43,23,51,35,69,32,66,31,32,21,62,24,45,27,38,30,17,17,21,56,50,55,55,72,38,44,51,20,64,34,74,44,23,34,50,47,50,34,51,56,33,43,60,22,23,54,39,32,20,30,56,34,32,59,19,29,24,41,32,51,18,29,25,31,34,66,41,62,58,47,19,32,25,60,22,20,42,18,28,40,52,37,35,45,54,59,39,18,46,24,22,61,56,17,29,28,39,30,20,18,54,49,33,50,57,45,41,23,39,29,46,18,68,62,43,35,21,61,24,21,36,69,55,21,18,26,46,23,48,36,42,41,52,27,36,34,38,21,33,31,61,46,55,19,39,61,28,28,61,61,54,50,34,64,29,41,45,48,34,40,45,51,24,57,34,36,56,45,38,29,54,40,29,35,28,58,46,57,22,28,17,23,19,38,39,47,39,22,36,32,29,43,51,42,34,51,38,21,34,32,38,27,35,29,40,46,38,37,44,25,38,50,46,20,47,46,34,61,36,22,33,43,55,20,38,30,21,60,43,36,57,25,22,23,28,49,37,38,27,57,22,38,40,44,57,41,50,33,29,28,35,36,30,35,27,22,24,47,47,18,31,45,39,30,30,36,20,61,27,59,60,35,40,20,47,40,20,75,47,34,23,29,17,39,34,20,52,22,35,40,51,34,24,41,64,23,29,28,33,35,42,51,37,45,52,63,72,35,35,70,65,53,49,60,18,21,42,32,25,47,38,34,52,23,29,45,45,30,45,26,23,43,42,57,24,27,54,42,17,48,41,57,44,49,42,18,18,25,41,39,63,27,39,20,70,46,47,35,24,22,60,53,34,37,35,29,23,31,18,38,52,23,48,46,26,47,65,31,68,42,49,39,31,28,62,21,60,30,41,25,27,40,58,27,31,50,45,33,62,38,42,49,53,45,38,25,51,38,29,36,57,24,29,67,20,21,20,42,19,34,43,34,33,47,41,40,28,62,28,54,22,18,52,41,21,46,57,22,33,17,45,30,40,31,42,53,37,46,32,17,42,37,29,54,38,46,24,57,59,33,53,26,55,44,30,24,67,56,26,52,38,31,53,27,22,37,20,60,59,37,43,17,45,18,20,34,17,30,18,22,39,38,38,34,28,41,61,75,50,49,26,39,24,56,17,37,57,42,22,39,41,43,35,34,33,49,69,41,63,75,34,69,45,43,26,35,48,57,21,28,51,34,41,26,40,44,41,41,55,43,51,31,30,31,18,33,34,40,40,29,29,43,19,35,28,34,61,47,31,18,29,59,50,49,60,35,48,21,41,56,26,23,58,61,29,22,43,31,90,22,55,28,38,60,34,72,57,39,50,45,37,27,20,50,22,46,22,59,57,26,52,47,19,26,42,39,26,50,46,18,26,32,47,25,42,53,23,23,32,29,40,45,55,59,56,23,37,25,22,47,38,27,25,22,33,28,29,30,65,28,39,35,25,56,46,56,51,17,30,41,35,27,25,53,26,41,28,48,22,40,52,40,57,53,31,20,50,34,37,52,29,49,47,51,23,55,28,30,23,19,22,64,21,75,29,47,20,33,45,23,38,67,21,38,31,59,36,50,60,30,28,36,27,44,56,51,48,31,35,35,36,47,23,51,46,31,27,56,48,29,18,37,51,31,29,39,48,44,39,22,42,27,60,36,45,40,63,41,34,49,43,36,27,52,29,27,24,22,30,28,20,45,43,53,74,39,33,49,21,65,39,49,47,46,29,44,37,45,51,26,54,20,29,24,36,60,42,42,27,27,52,31,50,59,39,23,38,31,24,46,38,39,44,46,30,24,62,25,45,21,64,39,60,35,27,41,32,24,32,42,23,31,19,61,39,25,58,27,51,43,67,50,20,57,42,67,47,30,51,24,70,35,61,26,64,39,23,33,31,28,50,51,34,37,29,24,44,24,54,30,39,70,41,44,42,29,30,59,21,26,57,59,87,25,39,53,38,17,23,30,51,54,60,28,39,55,33,40,30,40,18,49,42,28,32,35,23,27,27,41,31,64,45,22,69,40,45,47,34,48,39,28,41,49,24,35,22,42,34,32,54,38,22,42,31,57,44,42,24,38,29,38,49,51,21,29,54,52,23,43,45,51,73,61,40,32,35,42,46,33,47,33,28,29,33,33,57,40,36,52,24,28,48,36,55,27,22,29,28,26,72,59,37,27,39,25,18,25,36,52,20,25,26,48,39,30,26,36,49,34,28,28,42,26,54,66,61,45,27,53,57,41,45,43,29,65,31,52,18,57,63,29,32,34,39,28,28,40,25,54,48,67,24,25,41,51,41,27,41,51,34,43,31,34,43,52,37,60,21,34,40,26,46,58,32,90,38,20,43,36,36,30,45,51,29,35,56,64,62,33,22,44,37,34,30,50,25,34,50,72,60,41,71,52,30,40,31,44,38,46,57,39,43,44,35,18,25,44,39,61,19,45,26,61,36,62,36,38,54,31,26,50,52,46,32,30,58,36,31,52,29,48,30,58,20,19,51,54,26,44,80,55,43,49,27,42,39,36,63,49,36,46,47,39,33,43,30,27,80,31,34,28,55,42,67,43,19,51,21,42,28,82,40,35,24,27,40,27,34,18,34,44,39,58,49,61,27,59,41,35,36,21,38,59,59,38,22,47,59,37,34,35,55,18,48,27,32,62,18,50,38,18,46,26,55,40,29,60,45,43,47,47,50,22,49,41,68,44,32,49,47,35,37,57,45,36,33,27,32,53,44,43,47,37,43,50,25,40,39,41,32,26,30,34,42,45,22,25,22,46,48,45,19,31,30,40,60,25,51,32,51,42,39,24,58,43,31,28,31,61,46,62,23,46,53,53,50,26,44,36,35,70,57,40,18,45,52,24,42,28,56,32,59,21,32,49,51,33,23,34,44,55,57,59,26,45,37,21,36,37,65,59,32,25,40,21,49,48,42,53,29,54,66,39,62,30,25,26,45,26,26,45,20,48,33,44,32,54,40,56,47,46,24,37,49,44,21,55,48,38,48,39,52,50,47,33,24,39,31,42,22,34,23,38,45,19,37,59,36,35,25,41,40,34,44,31,41,43,58,43,50,44,46,52,52,34,23,68,41,41,67,27,25,28,51,33,53,35,31,22,55,45,37,28,37,50,51,40,47,44,30,41,47,26,34,47,21,45,33,33,31,32,35,43,49,42,31,51,51,39,30,27,32,37,47,31,42,60,42,38,24,34,29,36,50,17,54,26,54,21,24,41,50,32,49,61,65,45,59,34,24,42,37,36,21,34,41,18,26,23,34,37,54,32,42,48,28,31,28,24,26,37,24,23,49,75,74,26,18,33,44,53,60,28,36,45,23,34,46,58,63,41,45,41,90,41,22,53,49,51,59,28,25,62,28,62,32,47,60,55,18,43,31,22,24,42,53,42,48,33,33,30,31,26,61,45,57,33,36,35,20,61,36,41,25,37,19,66,23,30,53,25,51,17,61,44,38,32,46,36,30,33,36,85,48,58,45,37,55,39,28,34,36,22,35,49,21,41,32,51,32,61,60,42,82,26,18,57,25,34,71,35,47,33,38,50,39,20,40,66,36,57,46,33,58,30,32,22,34,54,22,30,71,72,43,65,43,32,43,32,53,22,40,22],\"x0\":\" \",\"xaxis\":\"x\",\"y0\":\" \",\"yaxis\":\"y\",\"type\":\"box\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"age\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0]},\"legend\":{\"tracegroupgap\":0},\"title\":{\"text\":\"Age analysis of men\"},\"boxmode\":\"group\",\"height\":300},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('4bca95c1-c7c6-42d4-889d-ff5f22f1efef');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# First I did value_counts(with normalize) for column 'education'\n",
        "# Then I chose only value for 'Bachelors'\n",
        "# Result rounded to second place\n",
        "percentage_bachelors = round((data.education.value_counts(normalize=True).Bachelors)*100,2) \n",
        "\n",
        "print(f\"Percentage with Bachelors degrees: {percentage_bachelors}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "I7q9LWBHZFUL",
        "outputId": "b7f672ee-bd93-4a32-8458-9e3d8ff68b62"
      },
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Percentage with Bachelors degrees: 16.45%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotly\n",
        "\n",
        "fig = px.pie(\n",
        "    data,\n",
        "    values=data[['education']].value_counts(),\n",
        "    names=data['education'].value_counts().index,\n",
        "    title=\"What is the percentage of people who have a Bachelor's degree?\"\n",
        ")\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 542
        },
        "id": "wrwhKN__ZHOz",
        "outputId": "c351d73d-7725-463b-8e75-f010c397ab03"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script src=\"https://cdn.plot.ly/plotly-2.18.2.min.js\"></script>                <div id=\"14dbd99d-e463-45d2-b249-03990e5cf453\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"14dbd99d-e463-45d2-b249-03990e5cf453\")) {                    Plotly.newPlot(                        \"14dbd99d-e463-45d2-b249-03990e5cf453\",                        [{\"domain\":{\"x\":[0.0,1.0],\"y\":[0.0,1.0]},\"hovertemplate\":\"label=%{label}<br>value=%{value}<extra></extra>\",\"labels\":[\"HS-grad\",\"Some-college\",\"Bachelors\",\"Masters\",\"Assoc-voc\",\"11th\",\"Assoc-acdm\",\"10th\",\"7th-8th\",\"Prof-school\",\"9th\",\"12th\",\"Doctorate\",\"5th-6th\",\"1st-4th\",\"Preschool\"],\"legendgroup\":\"\",\"name\":\"\",\"showlegend\":true,\"values\":[10501,7291,5355,1723,1382,1175,1067,933,646,576,514,433,413,333,168,51],\"type\":\"pie\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"legend\":{\"tracegroupgap\":0},\"title\":{\"text\":\"What is the percentage of people who have a Bachelor's degree?\"}},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('14dbd99d-e463-45d2-b249-03990e5cf453');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# First, I create mask, which extract rows with 'Bachelors', 'Masters', ans 'Doctorate'\n",
        "higher_education = ( (data['education'] == 'Doctorate')\n",
        "                    | (data['education'] == 'Bachelors') \n",
        "                    | (data['education'] == 'Masters'))\n",
        "\n",
        "# I used this mask and then I did value_counts(with normalize)\n",
        "# Next I chose only row with index '>50K' and rounded its value\n",
        "higher_education_rich = round((\n",
        "            data[higher_education].salary.value_counts(normalize=True)['>50K']\n",
        "                )*100 ,2)\n",
        "\n",
        "print(f\"Percentage of people with higher education that earn >50K: {higher_education_rich}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "XnlcWJJaZJt1",
        "outputId": "d9db821b-2e91-47a0-8603-8bacbff07386"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Percentage of people with higher education that earn >50K: 46.54%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Lower education is just not higher education\n",
        "lower_education = ~higher_education \n",
        "\n",
        "# The same as for higher education\n",
        "lower_education_rich = round((\n",
        "            data[lower_education].salary.value_counts(normalize=True)['>50K']\n",
        "                )*100 ,2)\n",
        "\n",
        "print(f\"Percentage without higher education that earn >50K: {lower_education_rich}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "uNW-TxKtZMMG",
        "outputId": "9b874cba-24ff-48d3-be5f-4c5aff51d51c"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Percentage without higher education that earn >50K: 17.37%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Seaborn\n",
        "\n",
        "# For this visualisation I had to rebuild dataset to other form\n",
        "\n",
        "# First let's do a copy\n",
        "data_visualisation = data.copy()\n",
        "\n",
        "# In new dataset we add column 'type-education' \n",
        "# with values \"Higher education\" and \"Lower education\"\n",
        "# based on a column 'education'\n",
        "data_visualisation['type-education'] = np.where(\n",
        "    ((data_visualisation['education'] == \"Doctorate\") \n",
        "    | (data_visualisation['education'] == \"Bachelors\") \n",
        "    | (data_visualisation['education'] == \"Masters\")),\n",
        "    \"Higher education\",\n",
        "    \"Lower education\"\n",
        ")\n",
        "\n",
        "# Abracadabra\n",
        "data_visualisation = (data_visualisation\n",
        "    # We group dataset by column 'type-education'\n",
        "    .groupby('type-education')['salary']\n",
        "    # At the same time, we count values with normalize \n",
        "    # In column 'salary', for each group\n",
        "    .value_counts(normalize=True)\n",
        "    # Now we have small grouped dataset with values from value_counts\n",
        "    # But now it will be difficult to visualize our dataset\n",
        "    # So we unsack column salary into two columns\n",
        "    .unstack()\n",
        "    # And then back to form with one column salary\n",
        "    # But now, values from value_counts is other column\n",
        "    .melt(ignore_index=False)\n",
        "    # And now we can reset index to have columns \n",
        "    # With salary and type education\n",
        "    # Insted of Multiindex with them\n",
        "    .reset_index())\n",
        "\n",
        "# Finally we can create chart\n",
        "sns.set_theme(style=\"whitegrid\")\n",
        "\n",
        "g = sns.catplot(\n",
        "    data=data_visualisation,\n",
        "    x='type-education',\n",
        "    y='value',\n",
        "    hue='salary',\n",
        "    kind='bar',\n",
        "    palette=\"dark\",\n",
        "    alpha=.6,\n",
        ")\n",
        "\n",
        "g.despine(left=True)\n",
        "g.set(title='Salary relative to education')\n",
        "g.set_axis_labels(\"Type of education\", \"Percentages\")\n",
        "g.legend.set_title(\"Salary\")\n",
        "\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 523
        },
        "id": "ZW1RVz3jZNtZ",
        "outputId": "e26ee742-3d0c-4882-bca9-092ae14ca64d"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 609.75x500 with 1 Axes>"
            ],
            "image/png": "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gFBYWCha319//SV4eXkJH3zwgUF7TEyM4OXlJRw7dkzf9v777wv+/v4P3WedTif07NlTGDt2rKDT6fTthYWFQteuXYUxY8bo29544w2hVatWwvXr1/Vt//zzj+Dj42Nw3DIyMgQvLy9h+/btRtvz8vIS4uLi9K+nT58utGjRQjh79myZtQmCIBQXFwtardZgWUZGhuDr6yskJCTo286ePSu63ffff1948cUX9a8PHjwoeHl5CZ9//rlBv8mTJwve3t7ClStXDGpu2bKlQdu992L9+vVG2yKiJxdvvRFVglQqRZ8+ffDjjz8iPz9f37579260bt0aTZo0MejfunVr+Pr66l83atQI3bp1w5EjR6DVaiEIAr7//nt07doVgiAgJydH/69Tp05Qq9U4f/68aD2HDx8GAIwZM8agfezYsQbLK+Ovv/7C5cuX0adPH+Tm5urruXPnDoKCgnDixAnodDpotVocOXIE3bt3R6NGjfTre3h4oFOnTpXeLgDodDocOnQIL774Ilq1amW0XCKRAABsbGwgld798aXVapGbmwt7e3s0b94cf/75Z5W2nZycDJlMhlGjRhm0jx07FoIgGN1W69ixI5o2bap/3aJFCzg6OiIjI6NK2yciy8Rbb0SV1K9fP3z55Zc4dOgQ+vXrh/T0dJw/fx5z58416vvMM88YtTVr1gyFhYXIycmBVCqFSqXC5s2bsXnz5jK3l5OTI1rL9evXIZVKDX5hA4CrqyvkcjmuX79eyb0DLl++DODufCwxarUaJSUlKCoqKnMfmzdvXqWQlpOTg/z8fDz77LPl9tPpdPjqq6+wYcMGXLt2DVqtVr/M2dm50tsF7h7LevXqwdHR0aD93iccHzyWDRs2NBpDoVBApVJVaftEZJkYlIgqydPTEy1btsTu3bvRr18/7N69G9bW1ggNDa30WDqdDgDw6quvon///mX2qcizne5daakOwv+fuDx9+nTRZw/Z29sbTBZ/GLH67g84lbF8+XIsXboUAwcOxFtvvQWFQgGpVIr58+c/to/8y2SyMtsf1/aJ6PFgUCKqgn79+iEmJga3bt3C3r170aVLFygUCqN+V65cMWq7fPky7Ozs4OLiAgBwcHCATqdDx44dK12Hm5sbdDodrly5YvBsp9u3b0OlUsHNza3SY967fejo6FhuTS4uLqhVq1aZ+/jvv/8avL53bB682nLjxg2jMR0dHXHp0qVyazxw4ADat2+P+fPnG7SrVCqDidSVCZBubm44duwY8vPzDa4qpaen65cT0dOHc5SIqqB3796QSCT4+OOPkZGRgVdffbXMfqdPnzaYY/Tff//hhx9+QHBwMGQyGWQyGXr16oUDBw7g4sWLRuuXd9sNADp37gwAWLdunUH72rVrDZZXhq+vL5o2bYo1a9agoKBAtCaZTIZOnTrh0KFDBoEnLS0NR44cMVjH0dERtWvXxm+//WbQvmHDBoPXUqkU3bt3x08//YQ//vjDaNv3rtbIZDKjKzf79+9HZmamQZudnR0A44BWlpCQEGi1WnzzzTcG7YmJiZBIJAgJCXnoGERU8/CKElEVuLi44IUXXkBSUhLkcjm6dOlSZj8vLy+MGzfO4PEAADB58mR9n3feeQfHjx/HkCFDMHjwYHh6ekKpVOL8+fM4duwYfv31V9E6WrRogf79+2Pz5s1QqVRo27Yt/vjjD+zcuRPdu3dHhw4dKr1vUqkU0dHRmDBhAnr37o0BAwagfv36yMzMxPHjx+Ho6Ijly5fr9+OXX37BiBEjMGzYMGi1Wnz99dfw9PTEhQsXDMYdPHgwVq5ciQ8++AC+vr747bffjK48AcC0adOQkpKCUaNGYciQIfDw8EBWVhaSkpKwYcMG/fFetmwZZsyYgdatW+PixYvYs2eP0WT6pk2bQi6XY9OmTXBwcIC9vT38/PyM+gFA165d0b59e3z22We4fv06vL29kZKSgh9++AHh4eFG88CI6OnAoERURX379sVPP/2E0NBQ0addt23bFv7+/li2bBlu3LgBT09PLFiwAC1atND3qVu3LrZu3Yply5bh4MGD2LhxI5ydneHp6Yl33333oXVER0ejcePG2LlzJw4dOoS6desiIiICb775ZpX3rX379ti8eTM+//xzfP3117hz5w5cXV3h5+eHoUOH6vu1aNECq1evxoIFCxAXF4cGDRpg8uTJyMrKMgpKkZGRyMnJwYEDB7B//36EhIRg1apVCAoKMuhXv359bNmyBUuXLsWePXuQn5+P+vXrIyQkBLVq1QIATJw4EYWFhdizZw/27duH5557DitWrMDixYsNxrK2tkZMTAw+/fRTzJkzBxqNBgsWLCgzKEmlUnzxxReIi4vDvn37sGPHDri5uWH69On6TxES0dNHInDmIVGVHDp0CJGRkfjmm28QGBhotNzb2xsjRozA7NmzzVAdERFVB85RIqqirVu3okmTJvonPhMRUc3DW29ElfTdd9/hwoUL+Pnnn/HBBx9U60fziYjIsjAoEVXStGnTYG9vj0GDBmH48OHmLoeIiB4hzlEiIiIiEsE5SkREREQiGJSIiIiIRDAoEREREYlgUCIiIiISwaBEREREJIJBiYiIiEgEgxIRERGRCAYlIiIiIhEMSkREREQi/h8n+i/UZhfaCAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Simply minimum value from column 'hours-per-week'\n",
        "min_work_hours = data['hours-per-week'].min()\n",
        "\n",
        "print(f\"Min work time: {min_work_hours} hours/week\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "WMOWBMrzZQys",
        "outputId": "ccbf7040-dcae-4419-cd82-92eaf4596411"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Min work time: 1 hours/week\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# First, I extract rows with the minimum number of hours  \n",
        "num_min_workers = data[data['hours-per-week'] == min_work_hours]\n",
        "\n",
        "# value_counts(Normalize=True) returns percentage of people who have >50K salary from above data \n",
        "# I choose only value from row with '>50K' and round it to second place\n",
        "rich_percentage = round((num_min_workers.salary.value_counts(normalize=True)['>50K'])*100,2)\n",
        "\n",
        "print(f\"Percentage of rich among those who work fewest hours: {rich_percentage}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "AqTnCZ0SZS-P",
        "outputId": "216af66f-4514-4a96-f40e-08001062717b"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Percentage of rich among those who work fewest hours: 10.0%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotly\n",
        "\n",
        "# I used variables from previous cells to extract dataset\n",
        "# only with people who works minimum number of hours\n",
        "# and next I count values in column 'salary' \n",
        "min_workers = round((num_min_workers.salary.value_counts(normalize=True))*100,2)\n",
        "\n",
        "# Now I can create the plot\n",
        "fig = px.bar(\n",
        "    min_workers, \n",
        "    x=min_workers.values,\n",
        "    color= min_workers.index,\n",
        "    title=\"Payout percentage distribution of people working the minimum number of hours\", \n",
        "    orientation='h', \n",
        "    height=325\n",
        ")\n",
        "\n",
        "fig.update_yaxes(domain=[0, 0.5])\n",
        "fig.update_xaxes(title='Percentages')\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 342
        },
        "id": "3_Z_3ZK3ZUHM",
        "outputId": "698f86f6-c5a5-40d0-c8c3-93941a3df5c7"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script src=\"https://cdn.plot.ly/plotly-2.18.2.min.js\"></script>                <div id=\"62c97fa1-feac-4941-8740-4bc620262f71\" class=\"plotly-graph-div\" style=\"height:325px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"62c97fa1-feac-4941-8740-4bc620262f71\")) {                    Plotly.newPlot(                        \"62c97fa1-feac-4941-8740-4bc620262f71\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"index=%{y}<br>x=%{x}<extra></extra>\",\"legendgroup\":\"<=50K\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"<=50K\",\"offsetgroup\":\"<=50K\",\"orientation\":\"h\",\"showlegend\":true,\"textposition\":\"auto\",\"x\":[90.0],\"xaxis\":\"x\",\"y\":[\"<=50K\"],\"yaxis\":\"y\",\"type\":\"bar\"},{\"alignmentgroup\":\"True\",\"hovertemplate\":\"index=%{y}<br>x=%{x}<extra></extra>\",\"legendgroup\":\">50K\",\"marker\":{\"color\":\"#EF553B\",\"pattern\":{\"shape\":\"\"}},\"name\":\">50K\",\"offsetgroup\":\">50K\",\"orientation\":\"h\",\"showlegend\":true,\"textposition\":\"auto\",\"x\":[10.0],\"xaxis\":\"x\",\"y\":[\">50K\"],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Percentages\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0,0.5],\"title\":{\"text\":\"index\"},\"categoryorder\":\"array\",\"categoryarray\":[\">50K\",\"<=50K\"]},\"legend\":{\"title\":{\"text\":\"index\"},\"tracegroupgap\":0},\"title\":{\"text\":\"Payout percentage distribution of people working the minimum number of hours\"},\"barmode\":\"relative\",\"height\":325},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('62c97fa1-feac-4941-8740-4bc620262f71');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "highest_earning_country = (data\n",
        "    # Grouping rows by countries\n",
        "    .groupby('native-country')['salary']\n",
        "    # For each country calculate their percentage of people who earn >50K   \n",
        "    .value_counts(normalize=True)[:,'>50K']\n",
        "    # Sort values from the biggest\n",
        "    .sort_values(ascending=False)\n",
        "    # Return name of country which is first\n",
        "    .index[0])\n",
        "\n",
        "print(\"Country with highest percentage of rich:\", highest_earning_country)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "czzBdMXDZVic",
        "outputId": "a8e58804-a1ef-4ee3-cf1e-06ae95c7bb14"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Country with highest percentage of rich: Iran\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# The same as above\n",
        "# But return value of that country\n",
        "# Rounded to second place\n",
        "highest_earning_country_percentage = (round((data\n",
        "    .groupby('native-country')['salary']\n",
        "    .value_counts(normalize=True)[:,'>50K']\n",
        "    .sort_values(ascending=False)[0])*100,2))\n",
        "\n",
        "print(f\"Highest percentage of rich people in country: {highest_earning_country_percentage}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "h6HGKlOkZXQ0",
        "outputId": "939b648f-a308-4b43-ca43-57f194b48659"
      },
      "execution_count": 24,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Highest percentage of rich people in country: 41.86%\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotly\n",
        "\n",
        "# First create dataset \n",
        "# The same as above but returns all countries\n",
        "data_visualisation = (round(data\n",
        "    .groupby('native-country')['salary']\n",
        "    .value_counts(normalize=True)[:,'>50K'] * 100,2))\n",
        "\n",
        "fig = px.scatter_geo(\n",
        "    data_visualisation, # Source of data\n",
        "    locations=data_visualisation.index,  # Locations which are interpreted\n",
        "    size=data_visualisation.values,  # Size of circle\n",
        "    hover_name=data_visualisation.index,  # Name displayed when hover circle\n",
        "    locationmode='country names', # Setting to interpreted location\n",
        "    color=data_visualisation.index, # Each country have self color\n",
        "    projection=\"natural earth\",  # Type of map\n",
        "    title='Countries and their percentage of the population earning >50K',\n",
        "    labels={\"color\":'Country', 'size':'Percentage'}\n",
        ")\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 542
        },
        "id": "NXbB2AdVZYfy",
        "outputId": "67cbbe7b-96a7-4e98-c712-c4a45f9c06f5"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script src=\"https://cdn.plot.ly/plotly-2.18.2.min.js\"></script>                <div id=\"f50806d3-46ea-44d8-b3d0-b106a342e9c3\" class=\"plotly-graph-div\" style=\"height:525px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"f50806d3-46ea-44d8-b3d0-b106a342e9c3\")) {                    Plotly.newPlot(                        \"f50806d3-46ea-44d8-b3d0-b106a342e9c3\",                        [{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"?\"],\"legendgroup\":\"?\",\"locationmode\":\"country names\",\"locations\":[\"?\"],\"marker\":{\"color\":\"#636efa\",\"size\":[25.04],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"?\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Cambodia\"],\"legendgroup\":\"Cambodia\",\"locationmode\":\"country names\",\"locations\":[\"Cambodia\"],\"marker\":{\"color\":\"#EF553B\",\"size\":[36.84],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Cambodia\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Canada\"],\"legendgroup\":\"Canada\",\"locationmode\":\"country names\",\"locations\":[\"Canada\"],\"marker\":{\"color\":\"#00cc96\",\"size\":[32.23],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Canada\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"China\"],\"legendgroup\":\"China\",\"locationmode\":\"country names\",\"locations\":[\"China\"],\"marker\":{\"color\":\"#ab63fa\",\"size\":[26.67],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"China\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Columbia\"],\"legendgroup\":\"Columbia\",\"locationmode\":\"country names\",\"locations\":[\"Columbia\"],\"marker\":{\"color\":\"#FFA15A\",\"size\":[3.39],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Columbia\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Cuba\"],\"legendgroup\":\"Cuba\",\"locationmode\":\"country names\",\"locations\":[\"Cuba\"],\"marker\":{\"color\":\"#19d3f3\",\"size\":[26.32],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Cuba\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Dominican-Republic\"],\"legendgroup\":\"Dominican-Republic\",\"locationmode\":\"country names\",\"locations\":[\"Dominican-Republic\"],\"marker\":{\"color\":\"#FF6692\",\"size\":[2.86],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Dominican-Republic\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Ecuador\"],\"legendgroup\":\"Ecuador\",\"locationmode\":\"country names\",\"locations\":[\"Ecuador\"],\"marker\":{\"color\":\"#B6E880\",\"size\":[14.29],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Ecuador\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"El-Salvador\"],\"legendgroup\":\"El-Salvador\",\"locationmode\":\"country names\",\"locations\":[\"El-Salvador\"],\"marker\":{\"color\":\"#FF97FF\",\"size\":[8.49],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"El-Salvador\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"England\"],\"legendgroup\":\"England\",\"locationmode\":\"country names\",\"locations\":[\"England\"],\"marker\":{\"color\":\"#FECB52\",\"size\":[33.33],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"England\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"France\"],\"legendgroup\":\"France\",\"locationmode\":\"country names\",\"locations\":[\"France\"],\"marker\":{\"color\":\"#636efa\",\"size\":[41.38],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"France\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Germany\"],\"legendgroup\":\"Germany\",\"locationmode\":\"country names\",\"locations\":[\"Germany\"],\"marker\":{\"color\":\"#EF553B\",\"size\":[32.12],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Germany\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Greece\"],\"legendgroup\":\"Greece\",\"locationmode\":\"country names\",\"locations\":[\"Greece\"],\"marker\":{\"color\":\"#00cc96\",\"size\":[27.59],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Greece\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Guatemala\"],\"legendgroup\":\"Guatemala\",\"locationmode\":\"country names\",\"locations\":[\"Guatemala\"],\"marker\":{\"color\":\"#ab63fa\",\"size\":[4.69],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Guatemala\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Haiti\"],\"legendgroup\":\"Haiti\",\"locationmode\":\"country names\",\"locations\":[\"Haiti\"],\"marker\":{\"color\":\"#FFA15A\",\"size\":[9.09],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Haiti\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Honduras\"],\"legendgroup\":\"Honduras\",\"locationmode\":\"country names\",\"locations\":[\"Honduras\"],\"marker\":{\"color\":\"#19d3f3\",\"size\":[7.69],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Honduras\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Hong\"],\"legendgroup\":\"Hong\",\"locationmode\":\"country names\",\"locations\":[\"Hong\"],\"marker\":{\"color\":\"#FF6692\",\"size\":[30.0],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Hong\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Hungary\"],\"legendgroup\":\"Hungary\",\"locationmode\":\"country names\",\"locations\":[\"Hungary\"],\"marker\":{\"color\":\"#B6E880\",\"size\":[23.08],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Hungary\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"India\"],\"legendgroup\":\"India\",\"locationmode\":\"country names\",\"locations\":[\"India\"],\"marker\":{\"color\":\"#FF97FF\",\"size\":[40.0],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"India\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Iran\"],\"legendgroup\":\"Iran\",\"locationmode\":\"country names\",\"locations\":[\"Iran\"],\"marker\":{\"color\":\"#FECB52\",\"size\":[41.86],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Iran\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Ireland\"],\"legendgroup\":\"Ireland\",\"locationmode\":\"country names\",\"locations\":[\"Ireland\"],\"marker\":{\"color\":\"#636efa\",\"size\":[20.83],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Ireland\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Italy\"],\"legendgroup\":\"Italy\",\"locationmode\":\"country names\",\"locations\":[\"Italy\"],\"marker\":{\"color\":\"#EF553B\",\"size\":[34.25],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Italy\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Jamaica\"],\"legendgroup\":\"Jamaica\",\"locationmode\":\"country names\",\"locations\":[\"Jamaica\"],\"marker\":{\"color\":\"#00cc96\",\"size\":[12.35],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Jamaica\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Japan\"],\"legendgroup\":\"Japan\",\"locationmode\":\"country names\",\"locations\":[\"Japan\"],\"marker\":{\"color\":\"#ab63fa\",\"size\":[38.71],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Japan\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Laos\"],\"legendgroup\":\"Laos\",\"locationmode\":\"country names\",\"locations\":[\"Laos\"],\"marker\":{\"color\":\"#FFA15A\",\"size\":[11.11],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Laos\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Mexico\"],\"legendgroup\":\"Mexico\",\"locationmode\":\"country names\",\"locations\":[\"Mexico\"],\"marker\":{\"color\":\"#19d3f3\",\"size\":[5.13],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Mexico\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Nicaragua\"],\"legendgroup\":\"Nicaragua\",\"locationmode\":\"country names\",\"locations\":[\"Nicaragua\"],\"marker\":{\"color\":\"#FF6692\",\"size\":[5.88],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Nicaragua\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Peru\"],\"legendgroup\":\"Peru\",\"locationmode\":\"country names\",\"locations\":[\"Peru\"],\"marker\":{\"color\":\"#B6E880\",\"size\":[6.45],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Peru\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Philippines\"],\"legendgroup\":\"Philippines\",\"locationmode\":\"country names\",\"locations\":[\"Philippines\"],\"marker\":{\"color\":\"#FF97FF\",\"size\":[30.81],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Philippines\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Poland\"],\"legendgroup\":\"Poland\",\"locationmode\":\"country names\",\"locations\":[\"Poland\"],\"marker\":{\"color\":\"#FECB52\",\"size\":[20.0],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Poland\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Portugal\"],\"legendgroup\":\"Portugal\",\"locationmode\":\"country names\",\"locations\":[\"Portugal\"],\"marker\":{\"color\":\"#636efa\",\"size\":[10.81],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Portugal\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Puerto-Rico\"],\"legendgroup\":\"Puerto-Rico\",\"locationmode\":\"country names\",\"locations\":[\"Puerto-Rico\"],\"marker\":{\"color\":\"#EF553B\",\"size\":[10.53],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Puerto-Rico\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Scotland\"],\"legendgroup\":\"Scotland\",\"locationmode\":\"country names\",\"locations\":[\"Scotland\"],\"marker\":{\"color\":\"#00cc96\",\"size\":[25.0],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Scotland\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"South\"],\"legendgroup\":\"South\",\"locationmode\":\"country names\",\"locations\":[\"South\"],\"marker\":{\"color\":\"#ab63fa\",\"size\":[20.0],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"South\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Taiwan\"],\"legendgroup\":\"Taiwan\",\"locationmode\":\"country names\",\"locations\":[\"Taiwan\"],\"marker\":{\"color\":\"#FFA15A\",\"size\":[39.22],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Taiwan\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Thailand\"],\"legendgroup\":\"Thailand\",\"locationmode\":\"country names\",\"locations\":[\"Thailand\"],\"marker\":{\"color\":\"#19d3f3\",\"size\":[16.67],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Thailand\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Trinadad&Tobago\"],\"legendgroup\":\"Trinadad&Tobago\",\"locationmode\":\"country names\",\"locations\":[\"Trinadad&Tobago\"],\"marker\":{\"color\":\"#FF6692\",\"size\":[10.53],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Trinadad&Tobago\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"United-States\"],\"legendgroup\":\"United-States\",\"locationmode\":\"country names\",\"locations\":[\"United-States\"],\"marker\":{\"color\":\"#B6E880\",\"size\":[24.58],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"United-States\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Vietnam\"],\"legendgroup\":\"Vietnam\",\"locationmode\":\"country names\",\"locations\":[\"Vietnam\"],\"marker\":{\"color\":\"#FF97FF\",\"size\":[7.46],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Vietnam\",\"showlegend\":true,\"type\":\"scattergeo\"},{\"geo\":\"geo\",\"hovertemplate\":\"<b>%{hovertext}</b><br><br>native-country=%{location}<br>Percentage=%{marker.size}<extra></extra>\",\"hovertext\":[\"Yugoslavia\"],\"legendgroup\":\"Yugoslavia\",\"locationmode\":\"country names\",\"locations\":[\"Yugoslavia\"],\"marker\":{\"color\":\"#FECB52\",\"size\":[37.5],\"sizemode\":\"area\",\"sizeref\":0.10464999999999999,\"symbol\":\"circle\"},\"mode\":\"markers\",\"name\":\"Yugoslavia\",\"showlegend\":true,\"type\":\"scattergeo\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"geo\":{\"domain\":{\"x\":[0.0,1.0],\"y\":[0.0,1.0]},\"projection\":{\"type\":\"natural earth\"},\"center\":{}},\"legend\":{\"title\":{\"text\":\"native-country\"},\"tracegroupgap\":0,\"itemsizing\":\"constant\"},\"title\":{\"text\":\"Countries and their percentage of the population earning >50K\"}},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('f50806d3-46ea-44d8-b3d0-b106a342e9c3');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Create a mask to choose only rows about people from India which earns >50K\n",
        "mask = (data['salary'] == '>50K') & (data['native-country'] == 'India')\n",
        "\n",
        "# Use mask to dataset\n",
        "top_IN_occupation = (data[mask]['occupation']\n",
        "    # Count values in column 'occupation'\n",
        "    .value_counts()[:1]\n",
        "    # Return occupation with highest number\n",
        "    .index[0])\n",
        "\n",
        "print(\"Top occupations in India:\", top_IN_occupation)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 37
        },
        "id": "LLEdQI6TZZ-a",
        "outputId": "ef1bf9d0-cb35-444b-9654-5a32dd440f62"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Top occupations in India: Prof-specialty\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Plotly\n",
        "\n",
        "# I used mask from previous cell to create dataset\n",
        "# Dataset includes occupations and their number\n",
        "data_visualisation = (data[mask]\n",
        "    .groupby(['occupation'], as_index=False)['age']\n",
        "    .count()\n",
        "    .rename({'age': 'count'}, axis=1))\n",
        "\n",
        "\n",
        "# Create plot\n",
        "fig = px.bar(\n",
        "    data_visualisation, \n",
        "    y=\"occupation\", \n",
        "    x=\"count\", \n",
        "    title=\"Professions of rich people in India\", \n",
        "    orientation='h', \n",
        "    height=400\n",
        ")\n",
        "\n",
        "fig.update_yaxes(title='Occupation')\n",
        "fig.update_xaxes(title='Count')\n",
        "\n",
        "fig.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 417
        },
        "id": "5Bkt3eycZcHD",
        "outputId": "5ad22b62-95f5-45af-8771-954944932607"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "  for (rule of document.styleSheets[0].cssRules){\n",
              "    if (rule.selectorText=='body') {\n",
              "      rule.style.fontSize = '16px'\n",
              "      break\n",
              "    }\n",
              "  }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<html>\n",
              "<head><meta charset=\"utf-8\" /></head>\n",
              "<body>\n",
              "    <div>            <script src=\"https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG\"></script><script type=\"text/javascript\">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}</script>                <script type=\"text/javascript\">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>\n",
              "        <script src=\"https://cdn.plot.ly/plotly-2.18.2.min.js\"></script>                <div id=\"58f55a82-a6dc-4e50-8a58-54733158844b\" class=\"plotly-graph-div\" style=\"height:400px; width:100%;\"></div>            <script type=\"text/javascript\">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById(\"58f55a82-a6dc-4e50-8a58-54733158844b\")) {                    Plotly.newPlot(                        \"58f55a82-a6dc-4e50-8a58-54733158844b\",                        [{\"alignmentgroup\":\"True\",\"hovertemplate\":\"count=%{x}<br>occupation=%{y}<extra></extra>\",\"legendgroup\":\"\",\"marker\":{\"color\":\"#636efa\",\"pattern\":{\"shape\":\"\"}},\"name\":\"\",\"offsetgroup\":\"\",\"orientation\":\"h\",\"showlegend\":false,\"textposition\":\"auto\",\"x\":[1,8,2,25,1,2,1],\"xaxis\":\"x\",\"y\":[\"Adm-clerical\",\"Exec-managerial\",\"Other-service\",\"Prof-specialty\",\"Sales\",\"Tech-support\",\"Transport-moving\"],\"yaxis\":\"y\",\"type\":\"bar\"}],                        {\"template\":{\"data\":{\"histogram2dcontour\":[{\"type\":\"histogram2dcontour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"choropleth\":[{\"type\":\"choropleth\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"histogram2d\":[{\"type\":\"histogram2d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmap\":[{\"type\":\"heatmap\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"heatmapgl\":[{\"type\":\"heatmapgl\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"contourcarpet\":[{\"type\":\"contourcarpet\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"contour\":[{\"type\":\"contour\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"surface\":[{\"type\":\"surface\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"},\"colorscale\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]]}],\"mesh3d\":[{\"type\":\"mesh3d\",\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}],\"scatter\":[{\"fillpattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2},\"type\":\"scatter\"}],\"parcoords\":[{\"type\":\"parcoords\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolargl\":[{\"type\":\"scatterpolargl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"bar\":[{\"error_x\":{\"color\":\"#2a3f5f\"},\"error_y\":{\"color\":\"#2a3f5f\"},\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"bar\"}],\"scattergeo\":[{\"type\":\"scattergeo\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterpolar\":[{\"type\":\"scatterpolar\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"histogram\":[{\"marker\":{\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"histogram\"}],\"scattergl\":[{\"type\":\"scattergl\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatter3d\":[{\"type\":\"scatter3d\",\"line\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattermapbox\":[{\"type\":\"scattermapbox\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scatterternary\":[{\"type\":\"scatterternary\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"scattercarpet\":[{\"type\":\"scattercarpet\",\"marker\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}}}],\"carpet\":[{\"aaxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"baxis\":{\"endlinecolor\":\"#2a3f5f\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"minorgridcolor\":\"white\",\"startlinecolor\":\"#2a3f5f\"},\"type\":\"carpet\"}],\"table\":[{\"cells\":{\"fill\":{\"color\":\"#EBF0F8\"},\"line\":{\"color\":\"white\"}},\"header\":{\"fill\":{\"color\":\"#C8D4E3\"},\"line\":{\"color\":\"white\"}},\"type\":\"table\"}],\"barpolar\":[{\"marker\":{\"line\":{\"color\":\"#E5ECF6\",\"width\":0.5},\"pattern\":{\"fillmode\":\"overlay\",\"size\":10,\"solidity\":0.2}},\"type\":\"barpolar\"}],\"pie\":[{\"automargin\":true,\"type\":\"pie\"}]},\"layout\":{\"autotypenumbers\":\"strict\",\"colorway\":[\"#636efa\",\"#EF553B\",\"#00cc96\",\"#ab63fa\",\"#FFA15A\",\"#19d3f3\",\"#FF6692\",\"#B6E880\",\"#FF97FF\",\"#FECB52\"],\"font\":{\"color\":\"#2a3f5f\"},\"hovermode\":\"closest\",\"hoverlabel\":{\"align\":\"left\"},\"paper_bgcolor\":\"white\",\"plot_bgcolor\":\"#E5ECF6\",\"polar\":{\"bgcolor\":\"#E5ECF6\",\"angularaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"radialaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"ternary\":{\"bgcolor\":\"#E5ECF6\",\"aaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"baxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"},\"caxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\"}},\"coloraxis\":{\"colorbar\":{\"outlinewidth\":0,\"ticks\":\"\"}},\"colorscale\":{\"sequential\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"sequentialminus\":[[0.0,\"#0d0887\"],[0.1111111111111111,\"#46039f\"],[0.2222222222222222,\"#7201a8\"],[0.3333333333333333,\"#9c179e\"],[0.4444444444444444,\"#bd3786\"],[0.5555555555555556,\"#d8576b\"],[0.6666666666666666,\"#ed7953\"],[0.7777777777777778,\"#fb9f3a\"],[0.8888888888888888,\"#fdca26\"],[1.0,\"#f0f921\"]],\"diverging\":[[0,\"#8e0152\"],[0.1,\"#c51b7d\"],[0.2,\"#de77ae\"],[0.3,\"#f1b6da\"],[0.4,\"#fde0ef\"],[0.5,\"#f7f7f7\"],[0.6,\"#e6f5d0\"],[0.7,\"#b8e186\"],[0.8,\"#7fbc41\"],[0.9,\"#4d9221\"],[1,\"#276419\"]]},\"xaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"yaxis\":{\"gridcolor\":\"white\",\"linecolor\":\"white\",\"ticks\":\"\",\"title\":{\"standoff\":15},\"zerolinecolor\":\"white\",\"automargin\":true,\"zerolinewidth\":2},\"scene\":{\"xaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"yaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2},\"zaxis\":{\"backgroundcolor\":\"#E5ECF6\",\"gridcolor\":\"white\",\"linecolor\":\"white\",\"showbackground\":true,\"ticks\":\"\",\"zerolinecolor\":\"white\",\"gridwidth\":2}},\"shapedefaults\":{\"line\":{\"color\":\"#2a3f5f\"}},\"annotationdefaults\":{\"arrowcolor\":\"#2a3f5f\",\"arrowhead\":0,\"arrowwidth\":1},\"geo\":{\"bgcolor\":\"white\",\"landcolor\":\"#E5ECF6\",\"subunitcolor\":\"white\",\"showland\":true,\"showlakes\":true,\"lakecolor\":\"white\"},\"title\":{\"x\":0.05},\"mapbox\":{\"style\":\"light\"}}},\"xaxis\":{\"anchor\":\"y\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Count\"}},\"yaxis\":{\"anchor\":\"x\",\"domain\":[0.0,1.0],\"title\":{\"text\":\"Occupation\"}},\"legend\":{\"tracegroupgap\":0},\"title\":{\"text\":\"Professions of rich people in India\"},\"barmode\":\"relative\",\"height\":400},                        {\"responsive\": true}                    ).then(function(){\n",
              "                            \n",
              "var gd = document.getElementById('58f55a82-a6dc-4e50-8a58-54733158844b');\n",
              "var x = new MutationObserver(function (mutations, observer) {{\n",
              "        var display = window.getComputedStyle(gd).display;\n",
              "        if (!display || display === 'none') {{\n",
              "            console.log([gd, 'removed!']);\n",
              "            Plotly.purge(gd);\n",
              "            observer.disconnect();\n",
              "        }}\n",
              "}});\n",
              "\n",
              "// Listen for the removal of the full notebook cells\n",
              "var notebookContainer = gd.closest('#notebook-container');\n",
              "if (notebookContainer) {{\n",
              "    x.observe(notebookContainer, {childList: true});\n",
              "}}\n",
              "\n",
              "// Listen for the clearing of the current output cell\n",
              "var outputEl = gd.closest('.output');\n",
              "if (outputEl) {{\n",
              "    x.observe(outputEl, {childList: true});\n",
              "}}\n",
              "\n",
              "                        })                };                            </script>        </div>\n",
              "</body>\n",
              "</html>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "1xPx4aCwZdzS"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}