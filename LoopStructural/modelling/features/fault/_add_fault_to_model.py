def add_fault_to_model(model, fault_surface_data,
        displacement,
        tol=None,
        fault_slip_vector=None,
        fault_center=None,
        major_axis=None,
        minor_axis=None,
        intermediate_axis=None,
        faultfunction="BaseFault",
        faults=[],
        **kwargs)


      if "fault_extent" in kwargs and major_axis is None:
            major_axis = kwargs["fault_extent"]
        if "fault_influence" in kwargs and minor_axis is None:
            minor_axis = kwargs["fault_influence"]
        if "fault_vectical_radius" in kwargs and intermediate_axis is None:
            intermediate_axis = kwargs["fault_vectical_radius"]

        logger.info(f'Creating fault "{fault_surface_data}"')
        logger.info(f"Displacement: {displacement}")
        logger.info(f"Tolerance: {tol}")
        logger.info(f"Fault function: {faultfunction}")
        logger.info(f"Fault slip vector: {fault_slip_vector}")
        logger.info(f"Fault center: {fault_center}")
        logger.info(f"Major axis: {major_axis}")
        logger.info(f"Minor axis: {minor_axis}")
        logger.info(f"Intermediate axis: {intermediate_axis}")
        fault_slip_vector = np.array(fault_slip_vector, dtype="float")
        fault_center = np.array(fault_center, dtype="float")

        for k, v in kwargs.items():
            logger.info(f"{k}: {v}")

        if tol is None:
            tol = model.tol
            # divide the tolerance by half of the minor axis, as this is the equivalent of the distance
            # of the unit vector
            # if minor_axis:
            # tol *= 0.1*minor_axis

        if displacement == 0:
            logger.warning(f"{fault_surface_data} displacement is 0")

        if "data_region" in kwargs:
            kwargs.pop("data_region")
            logger.error("kwarg data_region currently not supported, disabling")
        displacement_scaled = displacement / model.scale_factor
        # create fault frame
        interpolator = model.get_interpolator(**kwargs)
        # faults arent supported for surfe
        if not isinstance(interpolator, DiscreteInterpolator):
            logger.error(
                "Change interpolator to a discrete interpolation algorithm FDI/PLI"
            )
            interpolatortype = kwargs["interpolatortype"]
            raise InterpolatorError(f"Faults not supported for {interpolatortype}")
        fault_frame_builder = FaultBuilder(
            interpolator, name=fault_surface_data, model=self, **kwargs
        )
        model._add_faults(fault_frame_builder, features=faults)
        # add data
        fault_frame_data = model.data.loc[
            model.data["feature_name"] == fault_surface_data
        ]
        trace_mask = np.logical_and(
            fault_frame_data["coord"] == 0, fault_frame_data["val"] == 0
        )
        logger.info(f"There are {np.sum(trace_mask)} points on the fault trace")
        if np.sum(trace_mask) == 0:
            logger.error(
                "You cannot model a fault without defining the location of the fault"
            )
            raise ValueError(f"There are no points on the fault trace")

        mask = np.logical_and(
            fault_frame_data["coord"] == 0, ~np.isnan(fault_frame_data["gz"])
        )
        vector_data = fault_frame_data.loc[mask, ["gx", "gy", "gz"]].to_numpy()
        mask2 = np.logical_and(
            fault_frame_data["coord"] == 0, ~np.isnan(fault_frame_data["nz"])
        )
        vector_data = np.vstack(
            [vector_data, fault_frame_data.loc[mask2, ["nx", "ny", "nz"]].to_numpy()]
        )
        fault_normal_vector = np.mean(vector_data, axis=0)
        logger.info(f"Fault normal vector: {fault_normal_vector}")

        mask = np.logical_and(
            fault_frame_data["coord"] == 1, ~np.isnan(fault_frame_data["gz"])
        )
        if fault_slip_vector is None:
            if (
                "avgSlipDirEasting" in kwargs
                and "avgSlipDirNorthing" in kwargs
                and "avgSlipDirAltitude" in kwargs
            ):
                fault_slip_vector = np.array(
                    [
                        kwargs["avgSlipDirEasting"],
                        kwargs["avgSlipDirNorthing"],
                        kwargs["avgSlipDirAltitude"],
                    ],
                    dtype=float,
                )
            else:
                fault_slip_vector = (
                    fault_frame_data.loc[mask, ["gx", "gy", "gz"]]
                    .mean(axis=0)
                    .to_numpy()
                )
        if np.any(np.isnan(fault_slip_vector)):
            logger.info("Fault slip vector is nan, estimating from fault normal")
            strike_vector, dip_vector = get_vectors(fault_normal_vector[None, :])
            fault_slip_vector = dip_vector[:, 0]
            logger.info(f"Estimated fault slip vector: {fault_slip_vector}")

        if fault_center is not None and ~np.isnan(fault_center).any():
            fault_center = model.scale(fault_center, inplace=False)
        else:
            # if we haven't defined a fault centre take the
            #  center of mass for lines assocaited with the fault trace
            if (
                ~np.isnan(kwargs.get("centreEasting", np.nan))
                and ~np.isnan(kwargs.get("centreNorthing", np.nan))
                and ~np.isnan(kwargs.get("centreAltitude", np.nan))
            ):
                fault_center = model.scale(
                    np.array(
                        [
                            kwargs["centreEasting"],
                            kwargs["centreNorthing"],
                            kwargs["centreAltitude"],
                        ],
                        dtype=float,
                    ),
                    inplace=False,
                )
            else:
                mask = np.logical_and(
                    fault_frame_data["coord"] == 0, fault_frame_data["val"] == 0
                )
                fault_center = (
                    fault_frame_data.loc[mask, ["X", "Y", "Z"]].mean(axis=0).to_numpy()
                )
        if minor_axis:
            minor_axis = minor_axis / model.scale_factor
        if major_axis:
            major_axis = major_axis / model.scale_factor
        if intermediate_axis:
            intermediate_axis = intermediate_axis / model.scale_factor
        fault_frame_builder.create_data_from_geometry(
            fault_frame_data,
            fault_center,
            fault_normal_vector,
            fault_slip_vector,
            minor_axis=minor_axis,
            major_axis=major_axis,
            intermediate_axis=intermediate_axis,
            points=kwargs.get("points", False),
        )
        if "force_mesh_geometry" not in kwargs:

            fault_frame_builder.set_mesh_geometry(kwargs.get("fault_buffer", 0.2), 0)
        if "splay" in kwargs and "splayregion" in kwargs:
            fault_frame_builder.add_splay(kwargs["splay"], kwargs["splayregion"])

        kwargs["tol"] = tol
        fault_frame_builder.setup(**kwargs)
        fault = fault_frame_builder.frame
        fault.displacement = displacement_scaled
        fault.faultfunction = faultfunction

        for f in reversed(model.features):
            if f.type == FeatureType.UNCONFORMITY:
                fault.add_region(f)
                break
        if displacement == 0:
            fault.type = "fault_inactive"
        model._add_feature(fault)

        return fault