"""
This module provides classes for configure.
"""
import os
import overrides
import streamlit as st

import matplotlib.pyplot as plt
import lps_utils.utils as lps_utils
import lps_utils.quantities as lps_qty
import lps_synthesis.propagation.layers as lps_layer
import lps_synthesis.propagation.channel_description as lps_desc
import lps_synthesis.streamlit.base as lps_st

class SSP(lps_st.StElement):
    """ Class to represent and configure a SoundSpeedProfile. """

    def __init__(self):

        if "channel_description" not in st.session_state:
            self._reset_channel_state()

        if "file_to_save" not in st.session_state:
            st.session_state.file_to_save = None

    @st.dialog("Sobreescrever arquivo")
    def _override_dialog(self):
        st.text("O arquivo já existe. Deseja sobrescrevê-lo?")

        c0, c1 = st.columns([1, 1])
        with c0:
            if st.button("Sobreescrever"):
                if "file_to_save" in st.session_state:
                    st.session_state.channel_description.save(st.session_state.file_to_save)

                st.session_state.file_to_save = None
                st.rerun()

        with c1:
            if st.button("Cancelar"):
                st.session_state.file_to_save = None
                st.rerun()

    def _reset_channel_state(self) -> None:
        st.session_state.channel_description = lps_desc.Description()
        st.session_state.channel_description.add(lps_qty.Distance.m(0), lps_qty.Speed.m_s(1500))

    def _configure_channel_state(self) -> None:
        c0, c1, c2, c3 = st.columns([1, 2, 2, 1])
        with c0:
            st.text("Água")
        with c1:
            depth = st.number_input("Profundidade(m)", min_value=0, step=10)
        with c2:
            speed = st.number_input("Velocidade(m/s)", value=1500, min_value=1400,
                                    max_value=1600, step=10)
        with c3:
            if st.button("Add", key="Add_layer"):
                st.session_state.channel_description.add(lps_qty.Distance.m(depth),
                                                         lps_qty.Speed.m_s(speed))

        c0, c1, c2, c3 = st.columns([1, 2, 2, 1])
        with c0:
            st.text("Fundo")
        with c1:
            bottom_depth = st.number_input("Profundidade(m)", value=50, min_value=5, step=10)
        with c2:
            bottom_type = st.selectbox("Tipo", list(lps_layer.BottomType))
        with c3:
            if st.button("Add", key="Add_bottom"):
                st.session_state.channel_description.add(lps_qty.Distance.m(bottom_depth),
                                                         bottom_type)

        c0, c1, c2 = st.columns([1, 2, 1])
        with c0:
            st.text("Perfil:")
        with c1:
            filename = st.text_input("Nome do Arquivo:", label_visibility="collapsed",
                                     placeholder="nome sem extensão")
        with c2:
            if st.button("Salvar", disabled=not filename, key="save_ssp"):
                if filename and not filename.endswith(".json"):
                    filename += ".json"

                file_path = os.path.join(self._get_dir(), filename)

                if os.path.exists(file_path):
                    st.session_state.file_to_save = file_path
                else:
                    st.session_state.channel_description.save(file_path)

    def _show_channel_state(self) -> None:

        _, col2, _ = st.columns([1, 10, 1])
        with col2:
            st.divider()

        if st.checkbox("Exibir Camadas atuais"):
            for i, (d, s) in enumerate(st.session_state.channel_description):
                col1, col2 = st.columns([3, 1])

                if isinstance(s, lps_layer.BottomType):
                    col1.write(f"{s.__class__.__name__}: Profundidade {d}, Tipo {str(s)}")
                else:
                    col1.write(f"{s.__class__.__name__}: Profundidade {d}, " \
                            f"Velocidade {s.get_compressional_speed()}")

                if i == 0:
                    continue

                if col2.button("Remover", key=f"remove_{i}"):
                    st.session_state.channel_description.remove(d)
                    st.rerun()

    def _load_channel_state(self) -> None:
        files = lps_utils.find_files(current_directory=self._get_dir(), extension=".json")
        files = sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

        selected_file = st.selectbox(label="Perfil", options=files, index=None)

        if selected_file is not None:
            st.session_state.channel_description = lps_desc.Description.load(
                                    os.path.join(self._get_dir(), f"{selected_file}.json"))

    def _get_dir(self) -> str:
        path = os.path.join(lps_st.DEFAULT_DIR, "ssp")
        os.makedirs(path, exist_ok=True)
        return path

    @overrides.overrides
    def configure(self) -> lps_desc.Description:
        """Return a lps_desc.Description based on file and edition """

        options = ["Configurar", "Ler", "Ocultar"]
        opt = st.radio("Configuração do SSP", options, index=0,
                       key="Configuração do SSP", horizontal=True, label_visibility="collapsed")

        if opt == options[0]:
            self._configure_channel_state()
        elif opt == options[1]:
            self._load_channel_state()

        if opt != options[2]:
            self._show_channel_state()
        return st.session_state.channel_description

    @overrides.overrides
    def check_dialogs(self) -> None:
        """ Check the need for dialogs. """
        if st.session_state.file_to_save is not None:
            self._override_dialog()

    @overrides.overrides
    def plot(self):
        """Make a plot based on selected lps_desc.Description."""
        depths = []
        speeds = []
        for depth, layer in st.session_state.channel_description:
            if isinstance(layer, lps_layer.Water):
                depths.append(depth.get_m())
                speeds.append(layer.get_compressional_speed().get_m_s())

        plt.figure()
        plt.plot(speeds, depths, marker='o', linestyle='-')
        plt.gca().invert_yaxis()
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Depth (m)")
        plt.title("Sound Speed Profile")
        st.pyplot(plt)
